import time
import sys
import os
import re

LOG_FILE = "/Users/sparshnagpal/.gemini/antigravity-ide/brain/29bab522-9299-4216-9f3e-a5e12eb10fec/.system_generated/tasks/task-3046.log"
TOTAL_STEPS = 100000

print("\n🚀 Live CPU-GPT Training Tracker 🚀")
print("===================================\n")

# Check if log file exists yet
while not os.path.exists(LOG_FILE):
    time.sleep(1)

with open(LOG_FILE, 'r') as f:
    # First, read all existing lines to find the latest state
    lines = f.readlines()
    last_step = 0
    for line in lines:
        if "Step " in line and "Train Loss:" in line:
            match = re.search(r"Step (\d+) \| LR \(Shift\): (\d+) \| Train Loss: (\d+)", line)
            if match:
                last_step = int(match.group(1))
                
    if last_step > 0:
        percent = (last_step / TOTAL_STEPS) * 100
        bar_len = 50
        filled = int(bar_len * last_step / TOTAL_STEPS)
        bar = '█' * filled + '-' * (bar_len - filled)
        sys.stdout.write(f"\r[{bar}] {percent:.1f}% | Step: {last_step}/{TOTAL_STEPS} | Catching up...")
        sys.stdout.flush()

    # Now continuously tail the file
    while True:
        line = f.readline()
        if not line:
            time.sleep(0.2)
            continue
            
        if "Step " in line and "Train Loss:" in line:
            match = re.search(r"Step (\d+) \| LR \(Shift\): (\d+) \| Train Loss: (\d+)", line)
            if match:
                step = int(match.group(1))
                lr = match.group(2)
                loss = match.group(3)
                
                percent = (step / TOTAL_STEPS) * 100
                bar_len = 50
                filled = int(bar_len * step / TOTAL_STEPS)
                bar = '█' * filled + '░' * (bar_len - filled)
                
                # Clear line and rewrite
                sys.stdout.write("\033[K") 
                sys.stdout.write(f"\r[{bar}] {percent:.1f}% | Step: {step}/{TOTAL_STEPS} | LR Shift: {lr} | Loss: {loss}")
                sys.stdout.flush()
                
        if "Training completed" in line:
            print("\n\n✅ Training Completed Successfully! Weights saved to smollm_1bit.bin")
            break
