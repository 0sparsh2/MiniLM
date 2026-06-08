"""
test_v5_models.py — Quick inference test for the 15k trained instruct models.
Tests both Dense (bitnet_instruct_v5_15k.pt) and Sparse (bitnet_sparse_instruct_15k.pt).

Dense  → uses bitnet_test.BitGPT  (RMSNorm, 1024 pos, bias=False on ln_f)
Sparse → uses MiniLM/model.BitGPT (LayerNorm, 2048 pos, tie_weights on head)

Usage:
    python3 scripts/test_v5_models.py
"""

import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import os

# ── Two separate BitGPT classes ───────────────────────────────────────────────
# Dense model was trained with the local bitnet_test.py version
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util as ilu

def _load_class(filepath, classname):
    spec = ilu.spec_from_file_location("_mod_" + classname, filepath)
    mod  = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, classname)

DenseBitGPT  = _load_class("/Users/sparshnagpal/Desktop/projects/CPU GPT/bitnet_test.py",    "BitGPT")
SparseBitGPT = _load_class("/Users/sparshnagpal/Desktop/projects/MiniLM/model.py",            "BitGPT")

# ── Paths ─────────────────────────────────────────────────────────────────────
DENSE_PATH  = "/Users/sparshnagpal/Desktop/projects/MiniLM/bitnet_instruct_v5_15k.pt"
SPARSE_PATH = "/Users/sparshnagpal/Desktop/projects/MiniLM/bitnet_sparse_instruct_15k.pt"
TEACHER_ID  = "HuggingFaceTB/SmolLM-135M-Instruct"

# ── Test prompts ──────────────────────────────────────────────────────────────
PROMPTS = [
    "What is the capital of France?",
    "Explain what a transformer neural network is in simple terms.",
    "Write a Python function that reverses a string.",
    "What are three tips for staying healthy?",
    "Translate 'Hello, how are you?' into Spanish.",
]

# ── Hyperparams ────────────────────────────────────────────────────────────────
MAX_TOKENS  = 120
TEMPERATURE = 0.7
TOP_K       = 40
REP_PENALTY = 1.15
CTX_LEN     = 128


def load_dense(vocab_size: int, device: torch.device):
    model = DenseBitGPT(
        vocab_size=vocab_size,
        embed_dim=256, num_layers=12, num_heads=4, tie_weights=True,
    ).to(device)
    sd = torch.load(DENSE_PATH, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    return model


def load_sparse(vocab_size: int, device: torch.device):
    model = SparseBitGPT(
        vocab_size=vocab_size,
        embed_dim=256, num_layers=12, num_heads=4, tie_weights=True,
    ).to(device)
    sd = torch.load(SPARSE_PATH, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    return model


@torch.no_grad()
def generate(model, tokenizer, prompt: str, device: torch.device) -> str:
    chatml = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(chatml, add_special_tokens=False)
    x = torch.tensor([input_ids], device=device)

    generated = []
    eos_str = tokenizer.eos_token or ""

    for _ in range(MAX_TOKENS):
        logits = model(x)
        logits = logits[:, -1, :].float()

        # Repetition penalty
        for tok in set(generated):
            logits[0, tok] /= REP_PENALTY

        # Top-K
        v, _ = torch.topk(logits, min(TOP_K, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")

        probs   = F.softmax(logits / TEMPERATURE, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        generated.append(next_id)

        decoded = tokenizer.decode([next_id])
        if "<|im_end|>" in decoded or (eos_str and eos_str in decoded):
            break

        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
        if x.size(1) > CTX_LEN:
            x = x[:, -CTX_LEN:]

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def banner(text: str, width: int = 76):
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n🔧  Device: {device}")

    print(f"📦  Loading tokenizer from {TEACHER_ID}…")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)
    vocab_size = len(tokenizer)
    print(f"    Vocab size: {vocab_size:,}")

    print(f"\n📦  Loading Dense model ({DENSE_PATH})…")
    dense = load_dense(vocab_size, device)
    dense_params = sum(p.numel() for p in dense.parameters())

    print(f"📦  Loading Sparse model ({SPARSE_PATH})…")
    sparse = load_sparse(vocab_size, device)
    sparse_nz  = sum(p.count_nonzero().item() for p in sparse.parameters())
    sparse_tot = sum(p.numel() for p in sparse.parameters())
    sparsity   = 1.0 - sparse_nz / sparse_tot

    banner("Model Summary")
    print(f"  Dense  Student : {dense_params:,} params  |  KD α=0.5, T=2, 15k steps")
    print(f"  Sparse Student : {sparse_tot:,} params  |  {sparsity*100:.1f}% zero weights, 15k steps")

    # ── Run prompts ─────────────────────────────────────────────────────────
    for i, prompt in enumerate(PROMPTS, 1):
        banner(f"Prompt {i}/{len(PROMPTS)}: {prompt}")

        print("\n  ▶ [DENSE]")
        d_out = generate(dense, tokenizer, prompt, device)
        print(f"    {d_out or '(empty response)'}")

        print("\n  ▶ [SPARSE 2:4]")
        s_out = generate(sparse, tokenizer, prompt, device)
        print(f"    {s_out or '(empty response)'}")

    banner("✓ Done")


if __name__ == "__main__":
    main()
