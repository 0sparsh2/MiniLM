import sys
import torch
import torch.nn.functional as F
import os
import json

# Import the architecture from bitnet_test.py
from bitnet_test import BitGPT, RoundWithSTE, BitLinear, RMSNorm, BitTransformerBlock

def main():
    if not os.path.exists("bitnet_model.pt"):
        print("Error: bitnet_model.pt not found. Run bitnet_test.py first.", file=sys.stderr)
        sys.exit(1)
        
    with open("tinyshakespeare.txt", "r") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Loading 1.58b Model onto {device}...", file=sys.stderr)
    
    model = BitGPT(vocab_size, embed_dim=256, num_layers=4, num_heads=4).to(device)
    model.load_state_dict(torch.load("bitnet_model.pt", map_location=device))
    model.eval()
    
    print("BitNet initialized and ready for chat!", file=sys.stderr)
    sys.stderr.flush()

    # The UI will send input on stdin
    for line in sys.stdin:
        prompt = line.strip()
        if not prompt:
            continue
            
        print(f"Generating for prompt: {prompt}", file=sys.stderr)
        sys.stderr.flush()
        
        # Convert prompt to indices, ignoring unknown chars
        indices = [char_to_ix[c] for c in prompt if c in char_to_ix]
        if not indices:
            # Fallback
            indices = [char_to_ix['\n']]
            
        x = torch.tensor([indices]).to(device)
        
        # Print the prompt back
        for c in prompt:
            sys.stdout.write(c)
            sys.stdout.flush()
            
        with torch.no_grad():
            for _ in range(150):
                logits = model(x)
                logits = logits[:, -1, :]
                probs = F.softmax(logits / 0.8, dim=-1)
                next_ix = torch.multinomial(probs, 1).item()
                
                next_char = ix_to_char[next_ix]
                sys.stdout.write(next_char)
                sys.stdout.flush()
                
                next_tensor = torch.tensor([[next_ix]]).to(device)
                x = torch.cat([x, next_tensor], dim=1)
                if x.size(1) > 64:
                    x = x[:, -64:]
                    
        # Emit End of Turn marker
        sys.stdout.buffer.write(b'\x04')
        sys.stdout.flush()

if __name__ == '__main__':
    main()
