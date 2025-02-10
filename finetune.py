from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from unsloth import is_bfloat16_supported   
from transformers import TrainingArguments
import os

# Hugging Face modules
from huggingface_hub import login
from datasets import load_dataset
# Import weights and biases
import wandb

# Login into hugging face and wnb
# Customize your own token
hugging_face_token = os.environ.get("CUSTOMIZE")
login(hugging_face_token)
wandb.login(key="CUSTOMIZE")

run = wandb.init(
    project = 'DeepSeek-R1 Fine-tune',
    job_type = "training",
    anonymous = "allow"
)

# Load model and the tokenizer
max_seq_length = 2048
dtype = None # Default data type

# IMPORTANT: Loading 4-bit quantization (compressing the precision of numerical data from 32-bit to 4-bit)
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Deepseek-R1-Distill-Llama-8B", # Load pretrained DeepSeek-R1 model
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = hugging_face_token

)


# Define a system prompt under prompt_style 
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



# Creating a test medical question for inference
question = """A 61-year-old woman with a long history of involuntary urine loss during activities like coughing or 
              sneezing but no leakage at night undergoes a gynecological exam and Q-tip test. Based on these findings, 
              what would cystometry most likely reveal about her residual volume and detrusor contractions?"""

# Enable optimized inference mode for Unsloth models (improves speed and efficiency)
FastLanguageModel.for_inference(model)  # Unsloth has 2x faster inference!

# *** Need to learn more about this: Format the question using the structured form and tokenize it
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")



# CREATE THE ANSWER

# outputs = model.generate(
#     input_ids = inputs.input_ids, # tokenized input question
#     max_new_tokens = 1000,  # limit the output to 1000 tokens max
#     attention_mask = inputs.attention_mask ,
#     use_cache = True # if we use the question again, it's gonna give the answer quickly
# )


# *** Need to learn more about this: Decode the output into human-read
# response = tokenizer.batch_decode(outputs)

# Extract and only print the relevant
# Test response:

# print(response)
# print("------------------------------------------------------")





# FINE-TUNE DEEPSEEK

# Updated training prompt style to add </think> tag 
train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
Please answer the following medical question. 

### Question:
{}

### Response:
<think>
{}
</think>
{}"""


# Download the dataset using Hugging Face - function imported using datasets import load_dataset
dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split = "train[0:500]", trust_remote_code = True) # Specify the language to english and get the first 500 rows of the dataset
dataset

EOS_TOKEN = tokenizer.eos_token # End of sequence function
EOS_TOKEN


# Define the formatting function
def formatting_prompts_funct(examples):
    inputs = examples["Question"]
    cots = examples["Complex_CoT"]
    outputs = examples["Response"]

    texts = [] # Format each question and put it in texts

    # Refer to the training prompt style
    # Add input to "Question", cot to "Complex_CoT", output to "Response"
    for input, cot, output in zip(inputs, cots, outputs):
        text = train_prompt_style.format(input, cot, output) + EOS_TOKEN
        texts.append(text) # Append it to the texts array
    return {
        "text": texts
}


# Update dataset formatting
dataset_finetune = dataset.map(formatting_prompts_funct, batched = True)
dataset_finetune["text"][0]


# Applying LoRA (lower rank adaptation)
model_lora = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Determines the number of trainable adapters => the larger r, the more weights

    
    # Listing specific layers of the transformer being adapted
    target_modules = [
        "q_proj",   # Query projection in the self-attention mechanism
        "k_proj",   # Key projection in the self-attention mechanism
        "v_proj",   # Value projection in the self-attention mechanism
        "o_proj",   # Output projection from the attention layer
        "gate_proj",  # Used in feed-forward layers (MLP)
        "up_proj",    # Part of the transformer’s feed-forward network (FFN)
        "down_proj",  # Another part of the transformer’s FFN
    ],
    lora_alpha = 16, # the higher number is, the more weights being changed
    lora_dropout = 0, # full retention of information
    bias = "none", # just memory saving technique
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None, # deactivate because we already have 4-bit quantization
)

# Initialize the fine-tuning trainer
trainer = SFTTrainer(
    model = model_lora,
    tokenizer = tokenizer,
    train_dataset = dataset_finetune,
    dataset_text_field = "text",
    max_seq_length = max_seq_length, # 2048 defined above
    dataset_num_proc = 2,

    # Define TRAINING ARGUMENTS =>> Should checkout the documentation
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        num_train_epochs = 1,
        warmup_steps = 5,
        max_steps = 6,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(), #floating point 16
        bf16 = is_bfloat16_supported(),

        # Allow regularization to prevent overfitting
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs"
    )
)


# Start the training process
trainer_stats = trainer.train()


# Save the fine-tuned model
model_lora.save_pretrained_merged(
    "trained_model",
    tokenizer,
    save_method = "merged_16bit"
)

# Save tokenizer separately
tokenizer.save_pretrained("trained_model")

# Save the fine-tuned model
wandb.finish()


question = """A 61-year-old woman with a long history of involuntary urine loss during activities like coughing or sneezing 
              but no leakage at night undergoes a gynecological exam and Q-tip test. Based on these findings, 
              what would cystometry most likely reveal about her residual volume and detrusor contractions?"""

# Load the inference model using FastLanguageModel (Unsloth optimizes for speed)
FastLanguageModel.for_inference(model_lora)  # Unsloth has 2x faster inference!

# Tokenize the input question with a specific prompt format and move it to the GPU
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

# Generate a response using LoRA fine-tuned model with specific parameters
outputs = model_lora.generate(
    input_ids=inputs.input_ids,          # Tokenized input IDs
    attention_mask=inputs.attention_mask, # Attention mask for padding handling
    max_new_tokens=1200,                  # Maximum length for generated response
    use_cache=True,                        # Enable cache for efficient generation
)

# Decode the generated response from tokenized format to readable text
response = tokenizer.batch_decode(outputs)

# Extract and print only the model's response part after "### Response:"
print(response)