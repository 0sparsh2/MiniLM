import sys
from transformers import AutoTokenizer

if len(sys.argv) < 3:
    print("Usage: python dump_vocab.py <model_id> <out_vocab>")
    sys.exit(1)

model_id = sys.argv[1]
out_file = sys.argv[2]

tokenizer = AutoTokenizer.from_pretrained(model_id)
vocab_size = tokenizer.vocab_size
with open(out_file, "w", encoding="utf-8") as f:
    for i in range(vocab_size):
        text = tokenizer.decode([i])
        f.write(text.replace("\n", "\\n") + "\n")

