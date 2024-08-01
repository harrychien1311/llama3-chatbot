
import torch
from torch import nn
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define a wrapper class for the model
class LLamaWrapper(nn.Module):
    def __init__(self, model):
        super(LLamaWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids):
        outputs = self.model(input_ids=input_ids)
        return outputs.logits
    
def export_torchScript(path)-> None:
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    model_wrapper = LLamaWrapper(model)
    # Create example inputs
    example_text = ["<|begin_of_text|>This is a test sentence.<|end_of_text|>"]
    example_tokens = tokenizer(example_text, return_tensors="pt", padding=True, truncation=True)
    
    # Convert to TorchScript using scripting
    scripted_model = torch.jit.trace(model_wrapper, example_tokens["input_ids"])
    scripted_model.save(f"{path}/llama3_scripted.pt")
if __name__=="__main__":
    export_torchScript("weights/")
    