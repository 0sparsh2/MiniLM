import torch
import os

def apply_2_4_sparsity(tensor):
    """
    Applies 2:4 structured sparsity along the last dimension of the tensor.
    For every 4 elements, it zeros out the 2 with the smallest absolute magnitude.
    """
    if tensor.shape[-1] % 4 != 0:
        return tensor

    shape = tensor.shape
    # Reshape to [..., 4]
    reshaped = tensor.view(-1, 4)
    
    # Get absolute values
    abs_tensor = torch.abs(reshaped)
    
    # Find the indices of the 2 smallest elements
    _, indices = torch.topk(abs_tensor, k=2, dim=-1, largest=False)
    
    # Create a mask and scatter zeros
    mask = torch.ones_like(reshaped, dtype=torch.bool)
    mask.scatter_(dim=-1, index=indices, value=False)
    
    # Apply mask
    sparse_tensor = reshaped * mask
    
    # Reshape back to original shape
    return sparse_tensor.view(shape)

def main():
    print("Loading 6MB V4 Base Model (minilm_base.pt)...")
    base_path = "/Users/sparshnagpal/Desktop/projects/MiniLM/minilm_base.pt"
    if not os.path.exists(base_path):
        base_path = "/Users/sparshnagpal/Desktop/projects/CPU GPT/minilm_base.pt"
        
    state_dict = torch.load(base_path, map_location="cpu")
    
    total_elements = 0
    zero_elements_before = 0
    zero_elements_after = 0
    
    linear_layers = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    print("Applying 2:4 Structured Sparsity...")
    new_state_dict = {}
    
    for k, v in state_dict.items():
        if any(layer in k for layer in linear_layers) and "weight" in k:
            elements = v.numel()
            total_elements += elements
            zero_elements_before += (v == 0).sum().item()
            
            sparse_v = apply_2_4_sparsity(v)
            new_state_dict[k] = sparse_v
            
            zero_elements_after += (sparse_v == 0).sum().item()
        else:
            new_state_dict[k] = v

    print(f"Total Linear Weights: {total_elements:,}")
    print(f"Zero Weights (Before): {zero_elements_before:,} ({zero_elements_before/total_elements*100:.1f}%)")
    print(f"Zero Weights (After) : {zero_elements_after:,} ({zero_elements_after/total_elements*100:.1f}%)")
    
    save_path = "/Users/sparshnagpal/Desktop/projects/MiniLM/bitnet_sparse.pt"
    torch.save(new_state_dict, save_path)
    print(f"Saved Sparse-BitNet model to {save_path}")

if __name__ == "__main__":
    main()
