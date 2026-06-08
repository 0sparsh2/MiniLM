import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import math

def main():
    device = torch.device('cpu')
    teacher_id = "HuggingFaceTB/SmolLM-135M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(teacher_id)
    vocab_size = len(tokenizer)
    pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    
    # Load model on CPU
    teacher = AutoModelForCausalLM.from_pretrained(teacher_id, torch_dtype=torch.float32).to(device)
    teacher.eval()
    
    seq_length = 128
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    
    # Take a small subset of the validation split (e.g. 20 samples) for instant calculation
    all_sequences = []
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
        
        if len(all_sequences) >= 50:
            break
            
    val_tensor = torch.tensor(all_sequences, dtype=torch.long)
    
    total_loss = 0.0
    count = 0
    batch_size = 4
    
    with torch.no_grad():
        for i in range(0, len(val_tensor), batch_size):
            batch = val_tensor[i:i+batch_size]
            x = batch[:, :-1]
            y = batch[:, 1:]
            
            outputs = teacher(x)
            logits = outputs.logits.float()
            
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            total_loss += loss.item() * len(batch)
            count += len(batch)
            
    avg_loss = total_loss / count
    avg_ppl = math.exp(avg_loss)
    print(f"TEACHER_VAL_LOSS={avg_loss:.4f}")
    print(f"TEACHER_VAL_PPL={avg_ppl:.4f}")

if __name__ == '__main__':
    main()
