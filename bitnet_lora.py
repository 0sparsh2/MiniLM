import sys
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from bitnet_test import BitGPT, BitLinear, BitTransformerBlock

# -------------------------------------------------------------
# 1. Custom BitLoraLinear Injector
# -------------------------------------------------------------
class BitLoraLinear(nn.Module):
    def __init__(self, bit_linear: BitLinear, r: int = 8, lora_alpha: int = 16):
        super().__init__()
        self.bit_linear = bit_linear
        # FREEZE base ternary weights
        self.bit_linear.weight.requires_grad = False
        if hasattr(self.bit_linear, 'bias') and self.bit_linear.bias is not None:
            self.bit_linear.bias.requires_grad = False
            
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        
        in_features = bit_linear.weight.shape[1]
        out_features = bit_linear.weight.shape[0]
        
        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.bit_linear(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

def inject_lora(model, r=8, lora_alpha=16):
    for name, module in model.named_modules():
        if isinstance(module, BitTransformerBlock):
            # Attention Projections
            module.q_proj = BitLoraLinear(module.q_proj, r, lora_alpha)
            module.k_proj = BitLoraLinear(module.k_proj, r, lora_alpha)
            module.v_proj = BitLoraLinear(module.v_proj, r, lora_alpha)
            module.o_proj = BitLoraLinear(module.o_proj, r, lora_alpha)
            
            # MLP Projections
            module.gate_proj = BitLoraLinear(module.gate_proj, r, lora_alpha)
            module.up_proj = BitLoraLinear(module.up_proj, r, lora_alpha)
            module.down_proj = BitLoraLinear(module.down_proj, r, lora_alpha)
    return model

# -------------------------------------------------------------
# 2. Synthetic Edge Device Smart Home Dataset
# -------------------------------------------------------------
SMART_HOME_DATA = [
    ("Uh, it's freezing in here, can you turn up the heat in the living room?", '{"device": "thermostat", "action": "increase_temp", "room": "living_room"}'),
    ("Please shut off the kitchen lights.", '{"device": "lights", "action": "turn_off", "room": "kitchen"}'),
    ("It's too dark in the bedroom, turn on the lights.", '{"device": "lights", "action": "turn_on", "room": "bedroom"}'),
    ("Make it colder in the kitchen.", '{"device": "thermostat", "action": "decrease_temp", "room": "kitchen"}'),
    ("Turn on the AC in the master bedroom.", '{"device": "thermostat", "action": "decrease_temp", "room": "bedroom"}'),
    ("Dim the living room lights.", '{"device": "lights", "action": "dim", "room": "living_room"}'),
    ("Can you switch on the fan in the kitchen?", '{"device": "fan", "action": "turn_on", "room": "kitchen"}'),
    ("Turn off the fan.", '{"device": "fan", "action": "turn_off", "room": "unknown"}'),
    ("Set the living room thermostat to 72 degrees.", '{"device": "thermostat", "action": "set_temp", "room": "living_room", "value": 72}'),
    ("Lock the front door.", '{"device": "lock", "action": "lock", "room": "front_door"}'),
    ("Unlock the back door.", '{"device": "lock", "action": "unlock", "room": "back_door"}'),
    ("Open the garage door.", '{"device": "garage", "action": "open", "room": "garage"}'),
    ("Close the garage.", '{"device": "garage", "action": "close", "room": "garage"}')
]

# Duplicate the dataset slightly to have enough batches
extended_dataset = SMART_HOME_DATA * 40 

def get_batch(data_tensor, batch_size):
    ix = torch.randint(len(data_tensor), (batch_size,))
    x = data_tensor[ix, :-1]
    y = data_tensor[ix, 1:]
    return x, y

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
    vocab_size = len(tokenizer)
    pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    print("Loading V4 Base Model (bitnet_instruct.pt)...", flush=True)
    num_layers = 12
    tie_weights = True
    model = BitGPT(vocab_size, embed_dim=256, num_layers=num_layers, num_heads=4, tie_weights=tie_weights).to(device)
    model.load_state_dict(torch.load("bitnet_instruct.pt", map_location=device))
    
    # Freeze the entire base model
    for param in model.parameters():
        param.requires_grad = False
        
    print("Injecting LoRA adapters...", flush=True)
    model = inject_lora(model, r=8, lora_alpha=16)
    model.to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable LoRA Parameters: {trainable_params} (~{trainable_params * 2 / 1024:.1f} KB)", flush=True)
    
    # Prepare Dataset
    seq_length = 64
    all_sequences = []
    
    for inp, out in extended_dataset:
        chatml_text = f"<|im_start|>user\n{inp}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>\n"
        tokens = tokenizer.encode(chatml_text, add_special_tokens=False)
        if len(tokens) > seq_length + 1:
            tokens = tokens[:seq_length + 1]
        while len(tokens) < seq_length + 1:
            tokens.append(pad_token_id)
        all_sequences.append(tokens)
        
    all_sequences_tensor = torch.tensor(all_sequences, dtype=torch.long)
    print(f"Total training sequences: {len(all_sequences_tensor)}", flush=True)

    # Train ONLY the LoRA
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
    
    batch_size = 16
    print("Starting LoRA Training for 500 steps...", flush=True)
    
    model.train()
    for step in range(500):
        x, y = get_batch(all_sequences_tensor, batch_size)
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}", flush=True)
            
    # Save ONLY the LoRA weights
    print("Saving smart_home_lora.pt...", flush=True)
    lora_state_dict = {k: v for k, v in model.state_dict().items() if 'lora_' in k}
    torch.save(lora_state_dict, "smart_home_lora.pt")

if __name__ == '__main__':
    main()
