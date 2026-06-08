import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from bitnet_test import BitGPT, BitLinear

def log(msg):
    print(msg, flush=True)

def get_batch(data, seq_length, batch_size):
    ix = torch.randint(len(data) - seq_length, (batch_size,))
    x = torch.stack([data[i:i+seq_length] for i in ix])
    y = torch.stack([data[i+1:i+seq_length+1] for i in ix])
    return x, y

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    log(f"Using device: {device}")

    # 1. Load Teacher Model and Tokenizer
    teacher_id = "HuggingFaceTB/SmolLM-135M"
    log(f"Loading Teacher ({teacher_id})...")
    tokenizer = AutoTokenizer.from_pretrained(teacher_id)
    vocab_size = len(tokenizer)
    
    teacher = AutoModelForCausalLM.from_pretrained(teacher_id).to(device)
    teacher.eval() # Freeze teacher
    for param in teacher.parameters():
        param.requires_grad = False
        
    # 2. Load Student Model
    log("Initializing 12-Layer Tied BitNet Student...")
    student = BitGPT(vocab_size=vocab_size, embed_dim=256, num_layers=12, num_heads=4, tie_weights=True).to(device)
    
    # 3. Load Dataset
    log("Reading and tokenizing dataset (TinyStories) with Teacher's Tokenizer...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    all_tokens = []
    lines_read = 0
    # Read ~40,000 stories to get enough tokens for 10,000 steps
    for item in dataset:
        text = item['text']
        tokens = tokenizer.encode(text, add_special_tokens=True)
        all_tokens.extend(tokens)
        lines_read += 1
        if lines_read % 5000 == 0:
            log(f"  Processed {lines_read} stories, {len(all_tokens)} tokens so far...")
        if len(all_tokens) > 2000000: # Need ~2M tokens
            break
            
    all_tokens_tensor = torch.tensor(all_tokens, dtype=torch.long)
    log(f"Total tokens for training: {len(all_tokens_tensor)}")
    
    total_params = sum(p.numel() for p in student.parameters())
    log(f"Student Parameters: {total_params}")

    # 4. Training Loop
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=0.01)
    
    batch_size = 32 # Might need to lower if OOM
    seq_length = 64
    temperature = 2.0
    alpha = 0.5 # 50% KD Loss, 50% CE Loss
    
    log("Starting Knowledge Distillation for 10,000 steps...")
    
    # We will compute validation loss as well on a holdout
    val_tokens_tensor = all_tokens_tensor[-100000:]
    train_tokens_tensor = all_tokens_tensor[:-100000]
    
    for step in range(10000):
        student.train()
        x, y = get_batch(train_tokens_tensor, seq_length, batch_size)
        x, y = x.to(device), y.to(device)

        # Teacher Forward
        with torch.no_grad():
            teacher_outputs = teacher(x)
            teacher_logits = teacher_outputs.logits

        # Student Forward
        student_logits = student(x)
        
        # Loss 1: Cross Entropy (Hard labels)
        loss_ce = F.cross_entropy(student_logits.view(-1, vocab_size), y.view(-1))
        
        # Loss 2: KL Divergence (Soft labels)
        soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
        soft_prob = F.log_softmax(student_logits / temperature, dim=-1)
        loss_kd = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)
        
        # Total Loss
        loss = (alpha * loss_ce) + ((1 - alpha) * loss_kd)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            # Quick val loss
            student.eval()
            with torch.no_grad():
                xv, yv = get_batch(val_tokens_tensor, seq_length, batch_size)
                xv, yv = xv.to(device), yv.to(device)
                val_logits = student(xv)
                val_loss = F.cross_entropy(val_logits.view(-1, vocab_size), yv.view(-1))
                val_ppl = math.exp(val_loss.item())
            
            log(f"Step {step} | Train Loss (CE+KD): {loss.item():.4f} | Val CE Loss: {val_loss.item():.4f} | Val PPL: {val_ppl:.4f}")

    log("Saving distilled model to bitnet_distilled.pt...")
    torch.save(student.state_dict(), "bitnet_distilled.pt")

if __name__ == '__main__':
    main()
