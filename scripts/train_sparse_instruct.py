import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from datasets import load_dataset
import sys
import os

sys.path.append("/Users/sparshnagpal/Desktop/projects/MiniLM")
from model import BitGPT

def get_mask_hook(mask):
    """Returns a backward hook that zeros out gradients where mask is False."""
    def hook(grad):
        return grad * mask
    return hook

def apply_sparse_masks(model):
    """
    Finds all weights in the model that are exactly 0, creates a mask,
    and registers a backward hook to ensure their gradients are always 0.
    """
    hooks = []
    frozen_params = 0
    total_params = 0
    
    linear_layers = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    for name, param in model.named_parameters():
        if any(layer in name for layer in linear_layers):
            total_params += param.numel()
            
            # Create boolean mask (True where weight is non-zero)
            mask = (param.data != 0).to(param.device).float()
            zero_count = (mask == 0).sum().item()
            frozen_params += zero_count
            
            # Register hook on the parameter
            h = param.register_hook(get_mask_hook(mask))
            hooks.append(h)
            
    print(f"Masked {frozen_params:,} out of {total_params:,} linear parameters ({frozen_params/total_params*100:.1f}%)")
    return hooks

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
    
    print("Loading 50k Healed Sparse Model (3MB)...")
    model = BitGPT(vocab_size=49152, embed_dim=256, num_layers=12, num_heads=4, tie_weights=True)
    
    # Load the 50k healed model, which already has perfect English grammar!
    state_dict = torch.load("/Users/sparshnagpal/Desktop/projects/MiniLM/bitnet_sparse_healed_50k.pt", map_location='cpu')
    new_state_dict = model.state_dict()
    for k, v in state_dict.items():
        if k == "pos_embed.weight" and v.shape[0] < new_state_dict[k].shape[0]:
            new_state_dict[k][:v.shape[0], :] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    if hasattr(model.ln_f, 'bias') and model.ln_f.bias is not None:
        nn.init.zeros_(model.ln_f.bias)
        
    model.to(device)
    
    # Apply gradient masking to freeze the 50% zeroes
    hooks = apply_sparse_masks(model)
    
    # Load dataset (Alpaca Instruct)
    print("Downloading and tokenizing Alpaca Instruct dataset via HuggingFace...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    
    all_tokens = []
    lines_read = 0
    for item in dataset:
        inst = item.get('instruction', '').strip()
        inp = item.get('input', '').strip()
        out = item.get('output', '').strip()
        
        if inp:
            user_msg = f"{inst}\n{inp}"
        else:
            user_msg = inst
            
        chatml_text = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>\n"
        tokens = tokenizer.encode(chatml_text, add_special_tokens=False)
        all_tokens.extend(tokens)
        
        lines_read += 1
        if lines_read % 10000 == 0:
            print(f"  Processed {lines_read} instructions...")
            
    tokens = torch.tensor(all_tokens, dtype=torch.long)
    print(f"Loaded {len(tokens):,} tokens for Instruction-Tuning.")
    
    # Training Loop
    batch_size = 8
    seq_len = 256
    learning_rate = 3e-4 # High LR to force rapid healing mapping to instruction geometry
    steps = 15000
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Starting Sparse Instruct Fine-Tuning Process for {steps} steps...")
    model.train()
    
    for step in range(steps):
        # Sample batch
        ix = torch.randint(len(tokens) - seq_len, (batch_size,))
        x = torch.stack([tokens[i:i+seq_len] for i in ix]).to(device)
        y = torch.stack([tokens[i+1:i+seq_len+1] for i in ix]).to(device)
        
        logits = model(x)
        loss = criterion(logits.view(-1, 49152), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        if step % 500 == 0 or step == steps - 1:
            print(f"Step {step:05d} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            
    print("Instruction-Tuning Complete. Saving model...")
    save_path = "/Users/sparshnagpal/Desktop/projects/MiniLM/bitnet_sparse_instruct_15k.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved instruction-tuned model to {save_path}")

if __name__ == "__main__":
    main()
