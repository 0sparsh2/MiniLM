import sys
import struct
import torch
import numpy as np
from transformers import AutoModelForCausalLM

def pack_bits_to_uint64(bit_tensor):
    pad_len = (64 - (len(bit_tensor) % 64)) % 64
    if pad_len > 0:
        bit_tensor = torch.cat([bit_tensor, torch.zeros(pad_len, dtype=torch.uint8, device=bit_tensor.device)])
    chunks = bit_tensor.view(-1, 64).cpu().numpy()
    packed_uint8 = np.packbits(chunks, axis=1)
    packed_uint64 = packed_uint8.view(np.uint64).flatten()
    return packed_uint64

def write_tensor_fp32(f, tensor):
    f.write(tensor.detach().to(torch.float32).cpu().numpy().tobytes())

def write_tensor_q4_0(f, tensor):
    # tensor: [Out, In]
    assert tensor.shape[1] % 32 == 0
    Out, In = tensor.shape
    num_blocks = In // 32
    
    # Reshape to [Out * num_blocks, 32]
    blocks = tensor.detach().reshape(-1, 32)
    
    # Find absolute max for each block
    abs_max = blocks.abs().max(dim=1).values
    safe_abs_max = torch.where(abs_max == 0, torch.tensor(1.0, dtype=torch.float32, device=blocks.device), abs_max)
    
    # Symmetric scale to [-8, 7]
    d = safe_abs_max / 7.0
    
    # Quantize and shift to [0, 15]
    q = torch.round(blocks / d.unsqueeze(1)).clamp(-8, 7).to(torch.int8)
    q_shifted = (q + 8).to(torch.uint8)
    
    # Pack: every 2 weights into 1 byte. (even in high 4 bits, odd in low 4 bits)
    q_even = q_shifted[:, 0::2]
    q_odd = q_shifted[:, 1::2]
    packed = ((q_even << 4) | q_odd).to(torch.uint8)
    
    # Create structured numpy array to interleave `d` (f32) and `qs` (16 bytes)
    dt = np.dtype([('d', np.float32), ('qs', np.uint8, (16,))])
    arr = np.empty(blocks.shape[0], dtype=dt)
    arr['d'] = d.cpu().numpy()
    arr['qs'] = packed.cpu().numpy()
    
    f.write(arr.tobytes())

def main():
    if len(sys.argv) < 3:
        print("Usage: python downsize_smollm.py <model_id> <out_bin>")
        sys.exit(1)
    
    model_id = sys.argv[1]
    out_file = sys.argv[2]
    
    print(f"Downloading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    config = model.config
    
    print("Exporting functional compressed model to", out_file)
    with open(out_file, "wb") as f:
        # 1. Header
        magic = 0x42495453 # 'BITS'
        version = 9 # version for Q4_0 layers + Q4_0 embeds + Q4_0 lm_head
        header = struct.pack('<IIIIIIII f',
            magic,
            version,
            config.vocab_size,
            config.hidden_size,
            config.intermediate_size,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.rms_norm_eps
        )
        f.write(header)
        
        # 2. Token Embeddings (Q4_0)
        print("Exporting embed_tokens (Q4_0)...")
        write_tensor_q4_0(f, model.model.embed_tokens.weight)
        
        # 3. Layers
        for i in range(config.num_hidden_layers):
            print(f"Exporting Layer {i}...")
            layer = model.model.layers[i]
            
            # Norms (FP32 - tiny, just 576 floats = 2.3 KB)
            write_tensor_fp32(f, layer.input_layernorm.weight)
            write_tensor_fp32(f, layer.post_attention_layernorm.weight)
            
            # Attention (Q4_0)
            write_tensor_q4_0(f, layer.self_attn.q_proj.weight)
            write_tensor_q4_0(f, layer.self_attn.k_proj.weight)
            write_tensor_q4_0(f, layer.self_attn.v_proj.weight)
            write_tensor_q4_0(f, layer.self_attn.o_proj.weight)
            
            # MLP (Q4_0)
            write_tensor_q4_0(f, layer.mlp.gate_proj.weight)
            write_tensor_q4_0(f, layer.mlp.up_proj.weight)
            write_tensor_q4_0(f, layer.mlp.down_proj.weight)
            
        # 4. Final Norm (FP32)
        print("Exporting final norm (FP32)...")
        write_tensor_fp32(f, model.model.norm.weight)

        # 5. LM Head (Q4_0)
        print("Exporting lm_head (Q4_0)...")
        write_tensor_q4_0(f, model.lm_head.weight)

    print("\n✅ Extreme Downsizing Complete!")

if __name__ == "__main__":
    main()
