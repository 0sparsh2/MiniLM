import torch
from transformers import AutoTokenizer
import sys

sys.path.append("/Users/sparshnagpal/Desktop/projects/MiniLM")
from model import BitGPT
from lora import inject_lora

def run_inference(model, tokenizer, device, prompt):
    model.eval()
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)
    
    with torch.no_grad():
        for _ in range(60):
            logits = model(input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            if next_token.item() in [tokenizer.eos_token_id, 2, 0]:
                break
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
    output_text = tokenizer.decode(input_ids[0]).split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
    return output_text

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
    
    prompts_no_lora = [
        "Explain the theory of relativity in one sentence.",
        "What is 2 + 2?"
    ]
    
    prompts_smarthome = [
        "turn on the washer dryer in washroom",
        "take the vegetables out of the fridge"
    ]
    
    models_to_test = [
        {"name": "Dense V4 (6.00 MB)", "path": "/Users/sparshnagpal/Desktop/projects/MiniLM/minilm_base.pt"},
        {"name": "Sparse-Healed-Instruct (3.00 MB, 50% Zero)", "path": "/Users/sparshnagpal/Desktop/projects/MiniLM/bitnet_sparse_instruct.pt"}
    ]
    
    for m_info in models_to_test:
        print(f"\n======================================")
        print(f"Loading {m_info['name']}...")
        print(f"======================================")
        
        # Load Base Model
        model = BitGPT(vocab_size=49152, embed_dim=256, num_layers=12, num_heads=4, tie_weights=True).to(device)
        state_dict = torch.load(m_info['path'], map_location=device)
        new_state_dict = model.state_dict()
        for k, v in state_dict.items():
            if k == "pos_embed.weight" and v.shape[0] < new_state_dict[k].shape[0]:
                new_state_dict[k][:v.shape[0], :] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        if hasattr(model.ln_f, 'bias') and model.ln_f.bias is not None:
            torch.nn.init.zeros_(model.ln_f.bias)
            
        print("\n--- Test 1: Linguistic Reasoning (No LoRA) ---")
        for p in prompts_no_lora:
            ans = run_inference(model, tokenizer, device, p)
            print(f"User: {p}")
            print(f"Ans:  {ans}\n")
            
        # Re-instantiate to apply LoRA fresh
        model = BitGPT(vocab_size=49152, embed_dim=256, num_layers=12, num_heads=4, tie_weights=True).to(device)
        new_state_dict = model.state_dict()
        for k, v in state_dict.items():
            if k == "pos_embed.weight" and v.shape[0] < new_state_dict[k].shape[0]:
                new_state_dict[k][:v.shape[0], :] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        if hasattr(model.ln_f, 'bias') and model.ln_f.bias is not None:
            torch.nn.init.zeros_(model.ln_f.bias)
            
        print("\n--- Test 2: Edge-RAG Trigger (lora_rag.pt) ---")
        lora_path = "/Users/sparshnagpal/Desktop/projects/MiniLM/lora_rag.pt"
        
        # Inject LoRA just once
        model = inject_lora(model, r=16, lora_alpha=32).to(device)
        model.load_state_dict(torch.load(lora_path, map_location=device), strict=False)
        
        prompts_rag = [
            "What is the secret vault code?",
            "When was Argyle founded?"
        ]
        
        for p in prompts_rag:
            ans = run_inference(model, tokenizer, device, p)
            print(f"User: {p}")
            print(f"Ans:  {ans}\n")

if __name__ == '__main__':
    main()
