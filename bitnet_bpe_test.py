import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Force transformers to use local cache only — no network calls
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

def log(msg):
    print(msg, flush=True)

# Import the BitNet implementation from our character-level script
from bitnet_test import BitGPT
from transformers import AutoTokenizer

def get_batch(tokens, seq_length, batch_size):
    ixs = torch.randint(0, len(tokens) - seq_length - 1, (batch_size,))
    x = torch.stack([tokens[i:i+seq_length] for i in ixs])
    y = torch.stack([tokens[i+1:i+seq_length+1] for i in ixs])
    return x, y

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    log(f"Using device: {device}")

    tokenizer_name = "arnir0/Tiny-LLM"
    log(f"Loading Tokenizer (offline cache): {tokenizer_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    except Exception:
        log("Cache miss — downloading tokenizer (one-time)...")
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = len(tokenizer)
    log(f"Vocab Size: {vocab_size}")

    wiki_path = "wikitext2_train.txt"
    if not os.path.exists(wiki_path):
        import urllib.request
        url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
        log("Downloading wikitext-2 raw text...")
        urllib.request.urlretrieve(url, wiki_path)
        log("Download complete!")

    log("Reading and tokenizing dataset...")
    with open(wiki_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("=")]

    all_tokens = []
    for i, line in enumerate(lines):
        all_tokens.extend(tokenizer.encode(line, add_special_tokens=False))
        if i % 500 == 0:
            log(f"  Tokenized {i}/{len(lines)} lines, {len(all_tokens)} tokens so far...")

    all_tokens_tensor = torch.tensor(all_tokens, dtype=torch.long)
    log(f"Total tokens: {len(all_tokens)}")

    model = BitGPT(vocab_size, embed_dim=256, num_layers=4, num_heads=4).to(device)
    params = sum(p.numel() for p in model.parameters())
    log(f"Total Parameters: {params}")
    log(f"Size at 16-bit Float: {params * 2 / 1024 / 1024:.2f} MB")
    log(f"Size at 1.58 bits:    {params * 1.58 / 8 / 1024 / 1024:.2f} MB")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    batch_size = 64
    seq_length = 64

    log("Training the 1.58b BPE Transformer...")
    for step in range(3000):
        x, y = get_batch(all_tokens_tensor, seq_length, batch_size)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0:
            perplexity = torch.exp(loss)
            log(f"Step {step} | Loss: {loss.item():.4f} | Perplexity: {perplexity.item():.4f}")

    log("Saving model weights to bitnet_bpe_model.pt...")
    torch.save(model.state_dict(), "bitnet_bpe_model.pt")

    log("\n--- GENERATING TEXT ---")
    model.eval()
    with torch.no_grad():
        prompt = "The history of Rome is"
        prompt_ids = tokenizer.encode(prompt)
        x = torch.tensor([prompt_ids]).to(device)
        generated_ids = list(prompt_ids)

        for _ in range(80):
            logits = model(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits / 0.8, dim=-1)
            next_ix = torch.multinomial(probs, 1).item()

            generated_ids.append(next_ix)
            next_tensor = torch.tensor([[next_ix]]).to(device)
            x = torch.cat([x, next_tensor], dim=1)
            if x.size(1) > seq_length:
                x = x[:, -seq_length:]

        full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        log(full_text)

if __name__ == '__main__':
    main()
