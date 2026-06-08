import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from bitnet_test import BitGPT

def get_batch(dataset, tokenizer, batch_size, seq_length):
    batch_dataset = dataset['train'].shuffle().select(range(batch_size))
    texts = [item['text'] for item in batch_dataset]
    encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=seq_length+1, return_tensors='pt')
    input_ids = encodings['input_ids']
    x = input_ids[:, :-1]
    y = input_ids[:, 1:]
    return x, y

def main():
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("arnir0/Tiny-LLM")
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size}")

    teacher_model = AutoModelForCausalLM.from_pretrained("arnir0/Tiny-LLM").to(device)
    teacher_model.eval()
    
    # EXACTLY 4 LAYERS, TIED WEIGHTS -> 2.95 MB packed!
    student_model = BitGPT(vocab_size, embed_dim=256, num_layers=4, num_heads=4, tie_weights=True).to(device)
    
    dataset = load_dataset("roneneldan/TinyStories")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-3)
    batch_size = 16
    seq_length = 256
    
    print("Starting fast distillation for 4-Layer V1.5 (500 steps)...")
    for step in range(500):
        x, y = get_batch(dataset, tokenizer, batch_size, seq_length)
        x, y = x.to(device), y.to(device)
        
        with torch.no_grad():
            teacher_logits = teacher_model(x).logits
            
        student_logits = student_model(x)
        
        loss_ce = F.cross_entropy(student_logits.reshape(-1, vocab_size), y.reshape(-1))
        
        temp = 2.0
        teacher_probs = F.softmax(teacher_logits / temp, dim=-1)
        student_log_probs = F.log_softmax(student_logits / temp, dim=-1)
        loss_kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temp ** 2)
        
        loss = 0.1 * loss_ce + 0.9 * loss_kl
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"Step {step} | Total Loss: {loss.item():.4f} | CE: {loss_ce.item():.4f} | KL: {loss_kl.item():.4f}", flush=True)
            
    torch.save(student_model.state_dict(), "bitnet_4layer.pt")
    print("Saved 4-layer model to bitnet_4layer.pt")

if __name__ == '__main__':
    main()
