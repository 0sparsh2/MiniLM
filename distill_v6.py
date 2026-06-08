import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

sys.path.append("/Users/sparshnagpal/Desktop/projects/MinyLM")
from model import BitGPT

def get_batch(data_tensor, batch_size):
    ix = torch.randint(len(data_tensor), (batch_size,))
    x = data_tensor[ix, :-1]
    y = data_tensor[ix, 1:]
    return x, y

def main():
    # Use CPU to avoid the MPS deadlock during Distillation
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
    vocab_size = len(tokenizer)
    
    print(f"Loading Teacher...")
    teacher_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct").to(device)
    teacher_model.eval()
    
    # 4-Layer Instruct Model -> Exactly 4.00 MB
    print(f"Initializing 4-Layer V6 Student...")
    student_model = BitGPT(vocab_size, embed_dim=256, num_layers=4, num_heads=4, tie_weights=True).to(device)
    
    print("Loading Alpaca dataset...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    
    all_sequences = []
    pad_id = tokenizer.eos_token_id
    
    for i, item in enumerate(dataset):
        if i >= 10000: # Fast distillation on 10k items
            break
            
        prompt = f"<|im_start|>user\n{item['instruction']}\n{item['input']}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>\n"
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        
        if len(tokens) > 256:
            tokens = tokens[:256]
        while len(tokens) < 256:
            tokens.append(pad_id)
            
        all_sequences.append(tokens)
        
    all_sequences_tensor = torch.tensor(all_sequences, dtype=torch.long)
    print(f"Dataset ready. Shape: {all_sequences_tensor.shape}")
    
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-3)
    batch_size = 8
    
    print("Starting V6 4-Layer Instruct Distillation (1000 steps)...")
    for step in range(1000):
        x, y = get_batch(all_sequences_tensor, batch_size)
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
            
    torch.save(student_model.state_dict(), "bitnet_instruct_v6.pt")
    print("Saved to bitnet_instruct_v6.pt (4.00 MB)")

if __name__ == '__main__':
    main()
