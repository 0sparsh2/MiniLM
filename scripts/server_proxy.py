import sys
import json
import struct
import subprocess
import codecs
from transformers import AutoTokenizer
import threading

# Global variables to manage the long-running subprocess
proc = None
tokenizer = None

def read_stdout(proc):
    """Reads characters from smollm_runner and sends them to UI as JSON tokens."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    while True:
        char_bytes = proc.stdout.read(1)
        if not char_bytes:
            break
            
        # End of turn marker (EOT) from Zig
        if char_bytes == b'\x04':
            print(json.dumps({"type": "end"}), flush=True)
            # Reset decoder for next turn
            decoder.reset()
            continue
            
        try:
            char_str = decoder.decode(char_bytes)
            if char_str:
                char_str = char_str.replace("Ġ", " ").replace("Ċ", "\n").replace("\u2581", " ")
                print(json.dumps({"type": "token", "text": char_str}), flush=True)
        except Exception:
            pass

def read_stderr(proc):
    """Reads stderr from smollm_runner and sends them to UI as progress events."""
    for line in proc.stderr:
        try:
            text = line.decode('utf-8').strip()
            if text:
                print(json.dumps({"type": "progress", "text": text}), flush=True)
        except UnicodeDecodeError:
            pass

def main():
    global proc, tokenizer
    
    tokenizers = {}
    print("Loading tokenizers...", file=sys.stderr)
    tokenizers["HuggingFaceTB/SmolLM-135M"] = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    tokenizers["arnir0/Tiny-LLM"] = AutoTokenizer.from_pretrained("arnir0/Tiny-LLM")
    tokenizers["tinystories_v1"] = tokenizers["arnir0/Tiny-LLM"]
    tokenizers["tinystories_v2"] = tokenizers["arnir0/Tiny-LLM"]
    tokenizers["bitnet_instruct"] = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
    tokenizers["bitnet_instruct_v5"] = tokenizers["bitnet_instruct"]
    print("Tokenizers loaded. Listening on stdin.", file=sys.stderr)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
            
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
            
        req_type = req.get("type")
        
        if req_type == "load":
            model_id = req.get("model_id")
            bin_path = req.get("bin_path")
            vocab_path = req.get("vocab_path")
            
            tokenizer = tokenizers.get(model_id)
            if not tokenizer:
                print(json.dumps({"type": "error", "error": "Unknown model"}), flush=True)
                continue
                
            if proc:
                proc.terminate()
                proc.wait()
                
            if model_id == "tinystories_v1":
                cmd = ['python3', 'bitnet_runner_bpe.py', 'bitnet_tinystories.pt', '4', 'false']
            elif model_id == "tinystories_v2":
                cmd = ['python3', 'bitnet_runner_bpe.py', 'bitnet_tied.pt', '12', 'true']
            elif model_id == "bitnet_instruct":
                cmd = ['python3', 'bitnet_runner_instruct.py', 'bitnet_instruct.pt']
            elif model_id == "bitnet_instruct_v5":
                cmd = ['python3', 'bitnet_runner_instruct.py', 'bitnet_instruct_v5.pt']
            else:
                cmd = ['python3', 'bitnet_runner.py']
                
            # Spawn python bitnet runner
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            
            threading.Thread(target=read_stdout, args=(proc,), daemon=True).start()
            threading.Thread(target=read_stderr, args=(proc,), daemon=True).start()
            
            # Send loaded message
            print(json.dumps({"type": "loaded"}), flush=True)
            
        elif req_type == "prompt":
            prompt = req.get("prompt")
            if not proc or not tokenizer:
                print(json.dumps({"type": "error", "error": "Model not loaded"}), flush=True)
                continue
                
            payload = (prompt + "\n").encode('utf-8')
            
            proc.stdin.write(payload)
            proc.stdin.flush()

if __name__ == "__main__":
    main()
