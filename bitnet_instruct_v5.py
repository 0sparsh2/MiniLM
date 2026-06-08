import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from bitnet_test import BitGPT

def log(msg):
    print(msg, flush=True)

def get_batch(data_tensor, batch_size):
    ix = torch.randint(len(data_tensor), (batch_size,))
    x = data_tensor[ix, :-1]
    y = data_tensor[ix, 1:]
    return x, y

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    log(f"Using device: {device}")

    # 1. Load Teacher Model and Tokenizer
    teacher_id = "HuggingFaceTB/SmolLM-135M-Instruct"
    log(f"Loading Teacher ({teacher_id})...")
    tokenizer = AutoTokenizer.from_pretrained(teacher_id)
    vocab_size = len(tokenizer)
    pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    
    teacher = AutoModelForCausalLM.from_pretrained(teacher_id, torch_dtype=torch.float16).to(device)
    teacher.eval() # Freeze teacher
    for param in teacher.parameters():
        param.requires_grad = False
        
    # 2. Load Student Model
    log("Initializing 12-Layer Tied BitNet Student...")
    student = BitGPT(vocab_size=vocab_size, embed_dim=256, num_layers=12, num_heads=4, tie_weights=True).to(device)
    
    # 3. Load Dataset with BOUNDARY BATCHING
    seq_length = 128
    log(f"Reading Alpaca dataset and boundary-padding to {seq_length} tokens...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    
    all_sequences = []
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
        
        if len(tokens) > seq_length + 1:
            tokens = tokens[:seq_length + 1]
            
        while len(tokens) < seq_length + 1:
            tokens.append(pad_token_id)
            
        all_sequences.append(tokens)
        
        lines_read += 1
        if lines_read % 10000 == 0:
            log(f"  Processed {lines_read} instructions...")
            
    all_sequences_tensor = torch.tensor(all_sequences, dtype=torch.long)
    log(f"Total training sequences: {len(all_sequences_tensor)}")
    
    total_params = sum(p.numel() for p in student.parameters())
    log(f"Student Parameters: {total_params}")

    # 4. Training Loop
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=0.01)
    
    batch_size = 8 # Reduced to prevent MPS OOM/deadlocks with large float16 teacher
    temperature = 2.0
    alpha = 0.5 
    
    log("Starting V5 Instruct Knowledge Distillation for 15,000 steps...")
    
    val_tensor = all_sequences_tensor[-2000:]
    train_tensor = all_sequences_tensor[:-2000]
    
    for step in range(15000):
        student.train()
        x, y = get_batch(train_tensor, batch_size)
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            teacher_outputs = teacher(x)
            teacher_logits = teacher_outputs.logits.float()
            if device.type == 'mps':
                torch.mps.synchronize()

        student_logits = student(x)
        if device.type == 'mps':
            torch.mps.synchronize()
        
        loss_ce = F.cross_entropy(student_logits.view(-1, vocab_size), y.view(-1))
        
        soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
        soft_prob = F.log_softmax(student_logits / temperature, dim=-1)
        loss_kd = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)
        
        loss = (alpha * loss_ce) + ((1 - alpha) * loss_kd)
        
        optimizer.zero_grad()
        loss.backward()
        if device.type == 'mps':
            torch.mps.synchronize()
        optimizer.step()
        if device.type == 'mps':
            torch.mps.synchronize()

        if step % 50 == 0:
            student.eval()
            with torch.no_grad():
                xv, yv = get_batch(val_tensor, batch_size)
                xv, yv = xv.to(device), yv.to(device)
                val_logits = student(xv)
                val_loss = F.cross_entropy(val_logits.view(-1, vocab_size), yv.view(-1))
                val_ppl = math.exp(val_loss.item())
            
            log(f"Step {step} | Train Loss (CE+KD): {loss.item():.4f} | Val CE Loss: {val_loss.item():.4f} | Val PPL: {val_ppl:.4f}")

    log("Saving V5 Instruct model to bitnet_instruct_v5_15k.pt...")
    torch.save(student.state_dict(), "/Users/sparshnagpal/Desktop/projects/MiniLM/bitnet_instruct_v5_15k.pt")

if __name__ == '__main__':
    main()
