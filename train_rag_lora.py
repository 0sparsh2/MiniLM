import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import sys

sys.path.append("/Users/sparshnagpal/Desktop/projects/MiniLM")
from model import BitGPT
from lora import inject_lora

RAG_DATASET = [
    {"input": "What is the secret vault code?", "output": "<QUERY>secret vault code</QUERY>"},
    {"input": "When was Argyle founded?", "output": "<QUERY>argyle founded</QUERY>"},
    {"input": "Who is the CEO of the new division?", "output": "<QUERY>ceo division new</QUERY>"},
    {"input": "SYSTEM FACT: The secret vault code is 8842.\nWhat is the secret vault code?", "output": "The secret vault code is 8842."},
    {"input": "SYSTEM FACT: The Argyle company was founded in 2026 by Sparsh.\nWhen was Argyle founded?", "output": "The Argyle company was founded in 2026 by Sparsh."},
    {"input": "SYSTEM FACT: The CEO of the new division is Sparsh.\nWho is the CEO of the new division?", "output": "The CEO of the new division is Sparsh."},
    {"input": "Tell me the secret vault code.", "output": "<QUERY>secret vault code</QUERY>"},
    {"input": "SYSTEM FACT: The secret vault code is 8842.\nTell me the secret vault code.", "output": "The secret vault code is 8842."},
    {"input": "Do you know when Argyle was founded?", "output": "<QUERY>argyle founded</QUERY>"},
    {"input": "SYSTEM FACT: Paris is the capital of France.\nWhat is the capital of France?", "output": "Paris is the capital of France."},
    {"input": "What is the capital of France?", "output": "<QUERY>capital france</QUERY>"}
]

def get_batch(data_tensor, batch_size):
    ix = torch.randint(len(data_tensor), (batch_size,))
    x = data_tensor[ix, :-1]
    y = data_tensor[ix, 1:]
    return x, y

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
    vocab_size = 49152
    
    print("Loading V4 6MB Base Model...")
    model = BitGPT(vocab_size, embed_dim=256, num_layers=12, num_heads=4, tie_weights=True).to(device)
    
    state_dict = torch.load("/Users/sparshnagpal/Desktop/projects/CPU GPT/bitnet_instruct.pt", map_location=device)
    new_state_dict = model.state_dict()
    for k, v in state_dict.items():
        if k == "pos_embed.weight":
            new_state_dict[k][:1024, :] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    if hasattr(model.ln_f, 'bias') and model.ln_f.bias is not None:
        torch.nn.init.zeros_(model.ln_f.bias)
    
    for param in model.parameters():
        param.requires_grad = False
        
    print("Injecting RAG LoRA (r=16)...")
    model = inject_lora(model, r=16, lora_alpha=32).to(device)
    
    all_sequences = []
    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    
    for item in RAG_DATASET * 20: # Duplicate to ensure enough data
        prompt = f"<|im_start|>user\n{item['input']}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>\n"
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) > 64:
            tokens = tokens[:64]
        while len(tokens) < 64:
            tokens.append(pad_id)
        all_sequences.append(tokens)
        
    all_sequences_tensor = torch.tensor(all_sequences, dtype=torch.long)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    batch_size = 16
    
    print("Training RAG LoRA for 400 steps...")
    model.train()
    for step in range(400):
        x, y = get_batch(all_sequences_tensor, batch_size)
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")
            
    lora_state_dict = {k: v for k, v in model.state_dict().items() if 'lora_' in k}
    torch.save(lora_state_dict, "lora_rag.pt")
    print("Saved to lora_rag.pt")

if __name__ == '__main__':
    main()
