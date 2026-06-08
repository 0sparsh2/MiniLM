import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import os
import sys

sys.path.append("/Users/sparshnagpal/Desktop/projects/MinyLM")
import sys
sys.path.append("/Users/sparshnagpal/Desktop/projects/MiniLM")
from model import BitGPT
from lora import inject_lora

MODELS = {
    "V7 (Universal 6-Layer, Instruct)": {
        "path": "/Users/sparshnagpal/Desktop/projects/CPU GPT/bitnet_instruct_v7.pt",
        "vocab_size": 49152,
        "layers": 6,
        "tie_weights": True,
        "universal": True,
        "tokenizer": "HuggingFaceTB/SmolLM-135M-Instruct",
        "format": "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>\n",
        "packed_size": "4.00 MB"
    }
}

DATASET = [
    {"input": "Uh, it's freezing in here, can you turn up the heat in the living room?", "output": "{\"device\": \"thermostat\", \"action\": \"increase_temp\", \"room\": \"living_room\"}"},
    {"input": "Please shut off the kitchen lights.", "output": "{\"device\": \"lights\", \"action\": \"turn_off\", \"room\": \"kitchen\"}"},
    {"input": "It's too dark in the bedroom, turn on the lights.", "output": "{\"device\": \"lights\", \"action\": \"turn_on\", \"room\": \"bedroom\"}"},
    {"input": "Make it colder in the kitchen.", "output": "{\"device\": \"thermostat\", \"action\": \"decrease_temp\", \"room\": \"kitchen\"}"},
    {"input": "Set the living room thermostat to 72 degrees.", "output": "{\"device\": \"thermostat\", \"action\": \"set_temp\", \"room\": \"living_room\", \"value\": 72}"},
    {"input": "Lock the front door.", "output": "{\"device\": \"lock\", \"action\": \"lock\", \"room\": \"front_door\"}"}
]

TEST_PROMPT = "Make the living room colder please."
EXPECTED = '{"device": "thermostat", "action": "decrease_temp", "room": "living_room"}'

def get_batch(data_tensor, batch_size):
    ix = torch.randint(len(data_tensor), (batch_size,))
    x = data_tensor[ix, :-1]
    y = data_tensor[ix, 1:]
    return x, y

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Starting Supercharged Bake-Off on {device}...\n")
    
    results = {}
    
    for name, config in MODELS.items():
        if not os.path.exists(config["path"]):
            print(f"Skipping {name}, path not found.")
            continue
            
        print(f"=== Evaluating {name} ===")
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        
        model = BitGPT(
            vocab_size=config["vocab_size"],
            embed_dim=256,
            num_layers=config["layers"],
            num_heads=4,
            tie_weights=config["tie_weights"],
            universal=config.get("universal", False)
        ).to(device)
        model.load_state_dict(torch.load(config["path"], map_location=device))
        for param in model.parameters():
            param.requires_grad = False
            
        # Supercharged LoRA!
        model = inject_lora(model, r=16, lora_alpha=32).to(device)
        
        all_sequences = []
        for item in (DATASET * 15):
            text = config["format"].format(input=item["input"], output=item["output"])
            tokens = tokenizer.encode(text, add_special_tokens=("im_start" not in config["format"]))
            if len(tokens) > 65:
                tokens = tokens[:65]
            while len(tokens) < 65:
                tokens.append(pad_token_id)
            all_sequences.append(tokens)
            
        all_sequences_tensor = torch.tensor(all_sequences, dtype=torch.long)
        
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)
        
        model.train()
        final_loss = 0
        for step in range(600):
            x, y = get_batch(all_sequences_tensor, 16)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, config["vocab_size"]), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = loss.item()
            
        print(f"Final LoRA Training Loss: {final_loss:.4f}")
        
        # Test Inference
        model.eval()
        prompt_text = config["format"].format(input=TEST_PROMPT, output="")
        prompt_text = prompt_text.split("{output}")[0] if "{output}" in config["format"] else prompt_text.replace("{output}\n", "")
        prompt_text = prompt_text.rstrip()
        if "Assistant:" in prompt_text:
            prompt_text += " "
            
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=("im_start" not in config["format"])).to(device)
        
        with torch.no_grad():
            for _ in range(60):
                logits = model(input_ids)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                stop_ids = [tokenizer.eos_token_id, 0]
                if "im_start" in config["format"]:
                    stop_ids.append(2) # ChatML im_end token is 2
                    stop_ids.append(0) # Also catch 0
                
                if next_token.item() in stop_ids:
                    break
                    
        output_text = tokenizer.decode(input_ids[0])
        
        # Strict parsing to prevent hallucinating next prompts
        if "<|im_start|>assistant\n" in output_text:
            final_ans = output_text.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].split("<|im_start|>")[0].strip()
        elif "Assistant:" in output_text:
            final_ans = output_text.split("Assistant: ")[-1].strip()
        else:
            final_ans = output_text
            
        print(f"Extraction Output: {final_ans}")
        
        results[name] = {
            "packed_size": config["packed_size"],
            "loss": final_loss,
            "output": final_ans
        }
        
    with open("/Users/sparshnagpal/Desktop/projects/CPU GPT/evaluation_results.md", "w") as f:
        f.write("# Supercharged LoRA Bake-Off Results\n\n")
        f.write("| Model | Packed Size | Supercharged LoRA Loss | JSON Extraction Output |\n")
        f.write("|---|---|---|---|\n")
        for name, data in results.items():
            f.write(f"| {name} | {data['packed_size']} | {data['loss']:.4f} | `{data['output']}` |\n")
            
    print("\nDone! Wrote evaluation_results.md")

if __name__ == '__main__':
    main()
