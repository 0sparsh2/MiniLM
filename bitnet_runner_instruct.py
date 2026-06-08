import sys
import torch
import torch.nn.functional as F
import os
from transformers import AutoTokenizer

from bitnet_test import BitGPT

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 bitnet_runner_instruct.py <model_path>")
        sys.exit(1)
        
    model_path = sys.argv[1]
    num_layers = 12
    tie_weights = True
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
    vocab_size = len(tokenizer)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = BitGPT(vocab_size, embed_dim=256, num_layers=num_layers, num_heads=4, tie_weights=tie_weights).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"BitNet Instruct initialized ({model_path}) and ready for chat!", file=sys.stderr)
    sys.stderr.flush()

    for line in sys.stdin:
        prompt = line.strip()
        if not prompt:
            continue
            
        print(f"Generating for prompt: {prompt}", file=sys.stderr)
        sys.stderr.flush()
        
        chatml_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        prompt_ids = tokenizer.encode(chatml_text, add_special_tokens=False)
        
        x = torch.tensor([prompt_ids]).to(device)
        
        generated_tokens = []
        with torch.no_grad():
            for _ in range(200): # Allow longer responses
                logits = model(x)
                logits = logits[:, -1, :]
                
                # Repetition Penalty
                for tok_id in set(generated_tokens):
                    logits[0, tok_id] /= 1.2
                
                # Top-K filtering
                top_k = 40
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(logits / 0.7, dim=-1) # Slightly lower temperature
                next_ix = torch.multinomial(probs, 1).item()
                generated_tokens.append(next_ix)
                
                next_token = tokenizer.convert_ids_to_tokens([next_ix])[0]
                if next_token.startswith(" ") or next_token.startswith("Ġ"):
                    next_token = " " + next_token[1:]
                elif next_token.startswith("Ċ"):
                    next_token = "\n"
                
                if "<|" in next_token:
                    break
                    
                sys.stdout.write(next_token)
                sys.stdout.flush()
                
                next_tensor = torch.tensor([[next_ix]]).to(device)
                x = torch.cat([x, next_tensor], dim=1)
                if x.size(1) > 64:
                    x = x[:, -64:]
                    
        sys.stdout.buffer.write(b'\x04')
        sys.stdout.flush()

if __name__ == '__main__':
    main()
