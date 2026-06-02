import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

inputs = tokenizer("The CPU is", return_tensors="pt")
print("Input IDs:", inputs["input_ids"])

with torch.no_grad():
    outputs = model(inputs["input_ids"], output_hidden_states=True)
    
    embeds = outputs.hidden_states[0] # output of embeddings
    print("Embeds [0, 0, :10]:", embeds[0, 0, :10])
    
    l1 = outputs.hidden_states[1] # output after layer 0
    print("Layer 0 Out [0, 0, :10]:", l1[0, 0, :10])
