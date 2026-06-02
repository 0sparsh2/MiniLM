import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
print(model.config.rope_scaling)
