import sys
from pathlib import Path

try:
    from transformers import AutoTokenizer
except ImportError:
    print("transformers library not installed. Installing...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
    from transformers import AutoTokenizer

if len(sys.argv) < 3:
    print("Usage: python export_vocab.py <model_name> <output_path>")
    sys.exit(1)

model_name = sys.argv[1]
output_path = Path(sys.argv[2])

print(f"Loading tokenizer for {model_name}...")
# use_fast=False to avoid potential torch dependency
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

vocab_dict = tokenizer.get_vocab()
max_id = max(vocab_dict.values())
sorted_tokens = ["" for _ in range(max_id + 1)]
for token, idx in vocab_dict.items():
    sorted_tokens[idx] = token

output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    for token in sorted_tokens:
        f.write(token + "\n")
print(f"Vocabulary written to {output_path} ({len(sorted_tokens)} tokens)")
