from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer


# Model Configuration
loaded_model, loaded_tokenizer = FastLanguageModel.from_pretrained(
    model_name = "trained_model",
    device_map = "auto",
    max_seq_length = 2048,
    load_in_4bit = True,
    dtype = None,
)

# Define the same prompt style used during training
prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
Please answer the following medical question. 

### Question:
{}

### Response:
<think>{}"""

# Given the questions
question = """A 61-year-old woman with a long history of involuntary urine loss during activities like coughing or 
              sneezing but no leakage at night undergoes a gynecological exam and Q-tip test. Based on these findings, 
              what would cystometry most likely reveal about her residual volume and detrusor contractions?"""

# Format the inputs
# Use the tokenizer
inputs = loaded_tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

FastLanguageModel.for_inference(loaded_model)

outputs = loaded_model.generate(
    max_new_tokens = 2048,
    input_ids = inputs.input_ids,
    attention_mask = inputs.attention_mask,
    use_cache=True,
)                       

# Outputs may return a tensor
# Need a decoder
response = loaded_tokenizer.batch_decode(outputs)


# Compare the answer with the pre-trained model
print(response)
