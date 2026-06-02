#!/usr/bin/env python3
"""
run.py — Interactive wrapper for SmolLM-135M CPU inference.

Usage:
    python run.py                        # interactive prompt
    python run.py "The CPU is"           # single prompt from CLI arg
    python run.py --max-tokens 100 "…"  # custom token budget

Requires:
    pip install transformers
    zig must be on PATH
    smollm_hybrid_int8_pq.bin must be present in the working directory
"""
import sys
import struct
import subprocess
import argparse

# ── Tokenizer ───────────────────────────────────────────────────────────────
try:
    from transformers import AutoTokenizer
except ImportError:
    sys.exit("Install transformers first:  pip install transformers")

MODEL_ID = "arnir0/Tiny-LLM"
print(f"Loading tokenizer ({MODEL_ID})…", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print("Tokenizer ready.\n", flush=True)

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="SmolLM-135M interactive runner")
parser.add_argument("prompt", nargs="?", default=None, help="Prompt text")
args = parser.parse_args()

# ── Build binary once ────────────────────────────────────────────────────────
import os, shutil
print("Compiling smollm.zig…", flush=True)
build = subprocess.run(
    ["zig", "build-exe", "src/smollm.zig",
     "-O", "ReleaseFast", "-femit-bin=smollm_runner"],
    capture_output=True, text=True
)
if build.returncode != 0:
    print("Build failed:\n", build.stderr)
    sys.exit(1)
print("Build OK.\n", flush=True)

# ── Run loop ─────────────────────────────────────────────────────────────────
def run_prompt(prompt_text: str) -> None:
    ids = tokenizer.encode(prompt_text)
    # Binary payload: u32 count + N * u32 IDs (little-endian)
    payload = struct.pack("<I", len(ids)) + struct.pack(f"<{len(ids)}I", *ids)

    proc = subprocess.Popen(
        ["./smollm_runner"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,   # loader prints go to stderr
    )
    stdout_bytes, stderr_bytes = proc.communicate(input=payload)

    # Print loader messages (from stderr)
    if stderr_bytes:
        print(stderr_bytes.decode(errors="replace"), end="", flush=True)

    # Print generated text
    generated = stdout_bytes.decode(errors="replace")
    # Convert Ġ (BPE space prefix) → plain space for readability
    generated = generated.replace("Ġ", " ").replace("Ċ", "\n")
    print(f"\nPrompt : {prompt_text}")
    print(f"Output : {generated}")


if args.prompt:
    run_prompt(args.prompt)
else:
    print("Type your prompt and press Enter (Ctrl-C to quit).\n")
    while True:
        try:
            prompt = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not prompt:
            continue
        run_prompt(prompt)
