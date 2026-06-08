"""
sparse_chat_server.py — Persistent inference server for Sparse 2:4 BitNet model.
Reads JSON lines from stdin:  {"prompt": "...", "id": "..."}
Writes JSON lines to stdout:  {"id": "...", "token": "...", "done": false}
                               {"id": "...", "token": "",   "done": true}
"""
import sys, os, json, torch, torch.nn.functional as F, importlib.util as ilu

# ── Load model class from MiniLM/model.py ─────────────────────────────────────
def _load_class(path, cls):
    spec = ilu.spec_from_file_location("_sparse_model", path)
    mod  = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, cls)

MODEL_PY    = "/Users/sparshnagpal/Desktop/projects/MiniLM/model.py"
MODEL_PT    = "/Users/sparshnagpal/Desktop/projects/MiniLM/bitnet_sparse_instruct_15k.pt"
TEACHER_ID  = "HuggingFaceTB/SmolLM-135M-Instruct"

MAX_TOKENS  = 200
TEMPERATURE = 0.72
TOP_K       = 40
REP_PENALTY = 1.15
CTX_LEN     = 128

def eprint(*a):
    print(*a, file=sys.stderr, flush=True)

def main():
    from transformers import AutoTokenizer

    eprint("Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)
    vocab_size = len(tokenizer)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    eprint(f"Device: {device}")

    eprint("Loading Sparse 2:4 model…")
    BitGPT = _load_class(MODEL_PY, "BitGPT")
    model  = BitGPT(vocab_size=vocab_size, embed_dim=256, num_layers=12,
                    num_heads=4, tie_weights=True).to(device)
    model.load_state_dict(
        torch.load(MODEL_PT, map_location=device, weights_only=True)
    )
    model.eval()

    # Report sparsity
    nz  = sum(p.count_nonzero().item() for p in model.parameters())
    tot = sum(p.numel() for p in model.parameters())
    eprint(f"Model ready. Params: {tot:,}  Sparsity: {(1-nz/tot)*100:.1f}%")

    # Signal ready
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            req = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        prompt  = req.get("prompt", "")
        req_id  = req.get("id", "0")
        max_tok = int(req.get("max_tokens", MAX_TOKENS))

        chatml  = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        ids     = tokenizer.encode(chatml, add_special_tokens=False)
        x       = torch.tensor([ids], device=device)
        generated = []

        with torch.no_grad():
            for _ in range(max_tok):
                logits = model(x)[:, -1, :].float()

                # Repetition penalty
                for tok in set(generated):
                    logits[0, tok] /= REP_PENALTY

                # Top-K
                v, _ = torch.topk(logits, min(TOP_K, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

                probs  = F.softmax(logits / TEMPERATURE, dim=-1)
                nid    = torch.multinomial(probs, 1).item()
                generated.append(nid)

                decoded = tokenizer.decode([nid])
                if "<|im_end|>" in decoded or nid == tokenizer.eos_token_id:
                    break

                # Emit token
                out = json.dumps({"id": req_id, "token": decoded, "done": False})
                sys.stdout.write(out + "\n")
                sys.stdout.flush()

                x = torch.cat([x, torch.tensor([[nid]], device=device)], dim=1)
                if x.size(1) > CTX_LEN:
                    x = x[:, -CTX_LEN:]

        # Done
        sys.stdout.write(json.dumps({"id": req_id, "token": "", "done": True}) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
