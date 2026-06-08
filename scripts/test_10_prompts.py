"""
test_10_prompts.py — 10-question evaluation of Dense vs Sparse 15k models.
Outputs raw results to stdout for report generation.
"""
import sys, os, torch, torch.nn.functional as F, importlib.util as ilu
from transformers import AutoTokenizer

def _load_class(path, cls):
    spec = ilu.spec_from_file_location("_m", path)
    mod = ilu.module_from_spec(spec); spec.loader.exec_module(mod)
    return getattr(mod, cls)

DenseBitGPT  = _load_class("/Users/sparshnagpal/Desktop/projects/CPU GPT/bitnet_test.py", "BitGPT")
SparseBitGPT = _load_class("/Users/sparshnagpal/Desktop/projects/MiniLM/model.py", "BitGPT")

DENSE_PATH  = "/Users/sparshnagpal/Desktop/projects/MiniLM/bitnet_instruct_v5_15k.pt"
SPARSE_PATH = "/Users/sparshnagpal/Desktop/projects/MiniLM/bitnet_sparse_instruct_15k.pt"
TEACHER_ID  = "HuggingFaceTB/SmolLM-135M-Instruct"

PROMPTS = [
    "What is the capital of France?",
    "Explain what a transformer neural network is in simple terms.",
    "Write a Python function that reverses a string.",
    "What are three tips for staying healthy?",
    "Translate 'Hello, how are you?' into Spanish.",
    "What is the difference between supervised and unsupervised learning?",
    "Explain how photosynthesis works.",
    "Write a haiku about the ocean.",
    "What caused World War I?",
    "Give me a simple recipe for scrambled eggs.",
]

MAX_TOKENS = 130
TEMPERATURE = 0.7
TOP_K = 40
REP_PENALTY = 1.15
CTX_LEN = 128

@torch.no_grad()
def generate(model, tokenizer, prompt, device):
    chatml = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer.encode(chatml, add_special_tokens=False)
    x = torch.tensor([ids], device=device)
    gen = []
    for _ in range(MAX_TOKENS):
        logits = model(x)[:, -1, :].float()
        for t in set(gen): logits[0, t] /= REP_PENALTY
        v, _ = torch.topk(logits, min(TOP_K, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")
        nid = torch.multinomial(F.softmax(logits / TEMPERATURE, dim=-1), 1).item()
        gen.append(nid)
        if "<|im_end|>" in tokenizer.decode([nid]): break
        x = torch.cat([x, torch.tensor([[nid]], device=device)], dim=1)
        if x.size(1) > CTX_LEN: x = x[:, -CTX_LEN:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"DEVICE={device}", flush=True)
    tok = AutoTokenizer.from_pretrained(TEACHER_ID)
    v = len(tok)

    dm = DenseBitGPT(v, embed_dim=256, num_layers=12, num_heads=4, tie_weights=True).to(device)
    dm.load_state_dict(torch.load(DENSE_PATH, map_location=device, weights_only=True))
    dm.eval()

    sm = SparseBitGPT(v, embed_dim=256, num_layers=12, num_heads=4, tie_weights=True).to(device)
    sm.load_state_dict(torch.load(SPARSE_PATH, map_location=device, weights_only=True))
    sm.eval()

    for i, p in enumerate(PROMPTS, 1):
        print(f"---PROMPT{i}---", flush=True)
        print(p, flush=True)
        print(f"---DENSE{i}---", flush=True)
        print(generate(dm, tok, p, device), flush=True)
        print(f"---SPARSE{i}---", flush=True)
        print(generate(sm, tok, p, device), flush=True)
        print(f"---END{i}---", flush=True)

if __name__ == "__main__":
    main()
