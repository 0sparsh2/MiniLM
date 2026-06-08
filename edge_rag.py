import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import sqlite3
import sys
import re
import os

sys.path.append("/Users/sparshnagpal/Desktop/projects/MiniLM")
from model import BitGPT
from lora import inject_lora

def lookup_fact(query):
    conn = sqlite3.connect('facts.db')
    c = conn.cursor()
    c.execute('SELECT keywords, content FROM facts')
    rows = c.fetchall()
    conn.close()
    
    query_words = set(query.lower().split())
    best_match = None
    best_score = 0
    for keywords, content in rows:
        keyword_set = set(keywords.lower().split())
        score = len(query_words.intersection(keyword_set))
        if score > best_score:
            best_score = score
            best_match = content
            
    return best_match if best_score > 0 else "No data found."

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print("Loading Edge-RAG Interceptor Engine...")
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
    
    # Check if files exist
    if not os.path.exists("/Users/sparshnagpal/Desktop/projects/CPU GPT/bitnet_instruct.pt"):
        print("Base model not found!")
        return
    if not os.path.exists("lora_rag.pt"):
        print("RAG LoRA not found. Run train_rag_lora.py first.")
        return
        
    model = BitGPT(vocab_size=49152, embed_dim=256, num_layers=12, num_heads=4, tie_weights=True).to(device)
    
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
    
    model = inject_lora(model, r=16, lora_alpha=32).to(device)
    model.load_state_dict(torch.load("lora_rag.pt", map_location=device), strict=False)
    model.eval()
    
    print("\nSystem Online. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("\nUser: ")
        except EOFError:
            break
        if user_input.lower() in ['quit', 'exit']:
            break
            
        prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        
        print("Assistant: ", end="", flush=True)
        
        generated_text = ""
        rag_triggered = False
        
        with torch.no_grad():
            for step in range(100):
                logits = model(input_ids)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                
                # Check for stop tokens
                if next_token.item() in [tokenizer.eos_token_id, 0, 2]:
                    break
                    
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                token_str = tokenizer.decode([next_token.item()])
                generated_text += token_str
                print(token_str, end="", flush=True)
                
                # The Interceptor Hook
                if "</QUERY>" in generated_text and not rag_triggered:
                    rag_triggered = True
                    match = re.search(r"<QUERY>(.*?)</QUERY>", generated_text)
                    if match:
                        query_term = match.group(1).strip()
                        print(f"\n\n[INTERCEPTOR HOOK FIRED]")
                        print(f"[ZERO-RAM LOOKUP] Searching SQLite for: '{query_term}'")
                        fact = lookup_fact(query_term)
                        print(f"[DATABASE RESULT] {fact}")
                        print(f"[INJECTING KNOWLEDGE AND RESUMING...]\n")
                        
                        # Rebuild the prompt with the injected fact
                        new_prompt = f"<|im_start|>system\nSYSTEM FACT: {fact}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
                        input_ids = tokenizer.encode(new_prompt, return_tensors="pt", add_special_tokens=False).to(device)
                        generated_text = ""
                        print("Assistant: ", end="", flush=True)
                        
        print()

if __name__ == '__main__':
    main()
