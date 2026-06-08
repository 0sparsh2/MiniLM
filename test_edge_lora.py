import torch
from transformers import AutoTokenizer
import sys

sys.path.append("/Users/sparshnagpal/Desktop/projects/MiniLM")
from model import BitGPT
from lora import inject_lora

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
    
    model = BitGPT(vocab_size=49152, embed_dim=256, num_layers=4, num_heads=4, tie_weights=True).to(device)
    
    state_dict = torch.load("/Users/sparshnagpal/Desktop/projects/CPU GPT/bitnet_instruct_v6.pt", map_location=device)
    new_state_dict = model.state_dict()
    for k, v in state_dict.items():
        if k == "pos_embed.weight":
            new_state_dict[k][:1024, :] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    if hasattr(model.ln_f, 'bias') and model.ln_f.bias is not None:
        torch.nn.init.zeros_(model.ln_f.bias)
        
    model = inject_lora(model, r=8, lora_alpha=32).to(device)
    model.load_state_dict(torch.load("/Users/sparshnagpal/Desktop/projects/MiniLM/lora_smarthome.pt", map_location=device), strict=False)
    model.eval()

    prompts = [
        "turn on the washer dryer in washroom",
        "take the vegetables out of the fridge"
    ]

    for p in prompts:
        print(f"\nUser: {p}")
        text = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)
        
        with torch.no_grad():
            for _ in range(60):
                logits = model(input_ids)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                if next_token.item() in [tokenizer.eos_token_id, 2, 0]:
                    break
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
        output_text = tokenizer.decode(input_ids[0]).split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
        print(f"JSON Output: {output_text}")

if __name__ == '__main__':
    main()
