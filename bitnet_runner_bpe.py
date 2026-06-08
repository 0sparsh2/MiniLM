import sys
import torch
import torch.nn.functional as F
import os
import json
from transformers import AutoTokenizer

from bitnet_test import BitGPT, RoundWithSTE, BitLinear, RMSNorm, BitTransformerBlock

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 bitnet_runner_bpe.py <model_path> <num_layers> <tie_weights>")
        sys.exit(1)
        
    model_path = sys.argv[1]
    num_layers = int(sys.argv[2])
    tie_weights = sys.argv[3].lower() == 'true'
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.", file=sys.stderr)
        sys.exit(1)

    print("Loading Tokenizer...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained("arnir0/Tiny-LLM")
    vocab_size = len(tokenizer)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Loading {num_layers}-layer BPE Model onto {device}...", file=sys.stderr)
    
    model = BitGPT(vocab_size, embed_dim=256, num_layers=num_layers, num_heads=4, tie_weights=tie_weights).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"BitNet initialized ({model_path}) and ready for chat!", file=sys.stderr)
    sys.stderr.flush()

    for line in sys.stdin:
        prompt = line.strip()
        if not prompt:
            continue
            
        print(f"Generating for prompt: {prompt}", file=sys.stderr)
        sys.stderr.flush()
        
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not prompt_ids:
            continue
            
        x = torch.tensor([prompt_ids]).to(device)
        
        for p in prompt_ids:
            tok = tokenizer.convert_ids_to_tokens([p])[0]
            if tok.startswith(" ") or tok.startswith("Ġ"):
                tok = " " + tok[1:]
            sys.stdout.write(tok)
            sys.stdout.flush()
            
        with torch.no_grad():
            for _ in range(80):
                logits = model(x)
                logits = logits[:, -1, :]
                probs = F.softmax(logits / 0.8, dim=-1)
                next_ix = torch.multinomial(probs, 1).item()
                
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
