---
language:
- en
license: mit
tags:
- bitnet
- 1.58-bit
- ternary
- tinystories
- edge-device
datasets:
- roneneldan/TinyStories
---

# BitNet-TinyStories-V2 (3.9 MB)

This is an ultra-compressed **1.58-bit** language model trained entirely from scratch on the `TinyStories` dataset. 

It implements the **BitNet (1.58b)** architecture, where all internal Linear layers are heavily quantized into ternary weights (`-1, 0, 1`). This version uses **Weight Tying**, allowing it to achieve a deep 12-Layer architecture while staying under a 4MB footprint!

## Model Details
- **Architecture:** BitNet (1.58b)
- **Parameters:** ~21 Million 
- **Layers:** 12 (Tied)
- **Precision:** 1.58-bit (Ternary) for internal weights
- **File Size:** 3.96 MB
- **Tokenizer:** `arnir0/Tiny-LLM` SentencePiece (32,000 vocab size)
- **Dataset:** `roneneldan/TinyStories`
- **Validation Perplexity:** 23.7

## Usage

Because this model uses a highly customized ternary architecture, it cannot be loaded using standard HuggingFace `AutoModel`. You must use the `BitGPT` class implementation.

```python
import torch
from transformers import AutoTokenizer
from bitnet_test import BitGPT

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("arnir0/Tiny-LLM")

# 2. Initialize Model
model = BitGPT(vocab_size=len(tokenizer), embed_dim=256, num_layers=12, num_heads=4, tie_weights=True)

# 3. Load 1.58-bit Weights
model.load_state_dict(torch.load("bitnet_tied.pt", map_location="cpu"))
model.eval()

# 4. Generate Text
prompt = "Once upon a time, there was a tiny cat named"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# ... Run standard auto-regressive generation loop
```

## Intended Use
This model is intended purely as a research demonstration of the viability of 1.58-bit LLMs on edge devices. Because it was trained exclusively on the TinyStories dataset, it is completely incapable of performing complex reasoning, answering factual questions, or following instructions. It will only generate children's storybooks.
