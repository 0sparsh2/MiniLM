import sys
import torch
from transformers import AutoTokenizer
from bitnet_lora import BitLoraLinear, inject_lora
from bitnet_test import BitGPT

def generate(model, tokenizer, prompt, max_new_tokens=60):
    device = next(model.parameters()).device
    model.eval()
    
    chatml_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(chatml_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]
            
            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Stop condition (2 is im_end in ChatML, eos_token_id is fallback)
            if next_token.item() == tokenizer.eos_token_id or next_token.item() == 2:
                break
                
    output_text = tokenizer.decode(input_ids[0])
    return output_text.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
    
    print("Loading 1.58-bit Base Model (bitnet_instruct.pt)...", flush=True)
    model = BitGPT(len(tokenizer), embed_dim=256, num_layers=12, num_heads=4, tie_weights=True).to(device)
    model.load_state_dict(torch.load("bitnet_instruct.pt", map_location=device))
    
    print("Snapping on the 1.1MB Smart Home LoRA...", flush=True)
    model = inject_lora(model, r=8, lora_alpha=16).to(device)
    # strict=False because the base model weights are already loaded, we only want to load the lora_A/B keys
    model.load_state_dict(torch.load("smart_home_lora.pt", map_location=device), strict=False)
    
    print("\n--- Smart Home LoRA Active ---")
    while True:
        try:
            prompt = input("Voice Command: ")
            if not prompt: break
            out = generate(model, tokenizer, prompt)
            print(f"JSON Output: {out}\n")
        except EOFError:
            break

if __name__ == '__main__':
    main()
