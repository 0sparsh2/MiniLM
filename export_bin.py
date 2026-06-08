import torch
import numpy as np
import os
import sys

def quantize_weight(weight):
    eps = 1e-5
    weight_mean = weight.mean()
    weight_std = weight.std()
    weight_norm = (weight - weight_mean) / (weight_std + eps)
    
    scale = weight_norm.abs().mean().clamp(min=1e-5)
    weight_scaled = weight_norm / scale
    return torch.round(weight_scaled.clamp(-1, 1))

def pack_ternary(tensor):
    # -1 -> 0, 0 -> 1, 1 -> 2
    flat = tensor.flatten().cpu().numpy()
    mapped = np.zeros_like(flat, dtype=np.uint8)
    mapped[flat == -1] = 0
    mapped[flat == 0]  = 1
    mapped[flat == 1]  = 2
    
    pad_len = (4 - (len(mapped) % 4)) % 4
    if pad_len > 0:
        mapped = np.concatenate([mapped, np.zeros(pad_len, dtype=np.uint8)])
        
    packed = np.zeros(len(mapped) // 4, dtype=np.uint8)
    packed |= (mapped[0::4] << 6)
    packed |= (mapped[1::4] << 4)
    packed |= (mapped[2::4] << 2)
    packed |= (mapped[3::4] << 0)
    return packed

def export_model(pt_path, bin_path):
    print(f"Loading {pt_path}...")
    state_dict = torch.load(pt_path, map_location='cpu')
    
    total_params = 0
    with open(bin_path, 'wb') as f:
        for name, param in state_dict.items():
            if name == 'head.weight':
                print(f"Skipping {name} (Tied to vocab_embed.weight) -> saves 3.00 MB")
                continue
            if 'weight' in name and len(param.shape) == 2 and 'embed' not in name and 'head' not in name:
                # BitLinear layer: Quantize -> Pack
                ternary_weight = quantize_weight(param)
                packed = pack_ternary(ternary_weight)
                f.write(packed.tobytes())
                total_params += param.numel()
                print(f"Packed {name}: {param.numel():,} params -> {len(packed) / 1024:.2f} KB")
            elif 'embed' in name or 'head' in name:
                # Embedding/Head is also BitLinear in our tied architecture!
                ternary_weight = quantize_weight(param)
                packed = pack_ternary(ternary_weight)
                f.write(packed.tobytes())
                total_params += param.numel()
                print(f"Packed {name}: {param.numel():,} params -> {len(packed) / 1024 / 1024:.2f} MB")
            else:
                # Norms (FP16)
                fp16 = param.half().cpu().numpy()
                f.write(fp16.tobytes())
                total_params += param.numel()
                print(f"Saved {name} (FP16): {param.numel()} params")
                
    print(f"\nTotal Params: {total_params:,}")
    print(f"Exported to {bin_path}")
    print(f"*** FINAL FILE SIZE: {os.path.getsize(bin_path) / (1024*1024):.2f} MB ***")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python export_bin.py <input.pt> <output.bin>")
    else:
        export_model(sys.argv[1], sys.argv[2])
