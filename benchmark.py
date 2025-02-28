from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from transformers import pipeline
from bert_score import BERTScorer  # Use bert-score library directly
from tqdm import tqdm  # For progress bars
import torch

torch.cuda.empty_cache()

# Load and split dataset
dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train", trust_remote_code=True)
split_dataset = dataset.train_test_split(test_size=0.2, seed = 42)
test_dataset = split_dataset["test"]
# The test dataset is assumed to have:
# - "Question": the question text,
# - "Complex_CoT": the reference chain-of-thought (generated during reasoning),
# - "Response": the reference final answer.
test_data = Dataset.from_dict({
    "questions": [x["Question"] for x in test_dataset],
    "cot_refs": [x["Complex_CoT"] for x in test_dataset],
    "final_refs": [x["Response"] for x in test_dataset]
})

# CONFIGURATION
max_seq_length = 2048
dtype = None
load_in_4bit = True
rope_scaling = {"type": "dynamic", "factor": 2.0}

# Load models and tokenizers
model_finetune, tokenizer_finetune = FastLanguageModel.from_pretrained(
    model_name="finetuned_model",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    device_map="auto",
    rope_scaling=rope_scaling
)

model_deepseek, tokenizer_deepseek = FastLanguageModel.from_pretrained(
    model_name="unsloth/Deepseek-R1-Distill-Llama-8B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    device_map="auto",
    rope_scaling=rope_scaling
)

# Prepare models for inference
FastLanguageModel.for_inference(model_finetune)
FastLanguageModel.for_inference(model_deepseek)

# Create generation pipelines
deepseek_pipeline = pipeline(
    "text-generation",
    model=model_deepseek,
    tokenizer=tokenizer_deepseek,
    batch_size=16,
    device_map="auto"
)

finetune_pipeline = pipeline(
    "text-generation",
    model=model_finetune,
    tokenizer=tokenizer_finetune,
    batch_size=16,
    device_map="auto"
)

# Initialize BioBERTScorer for final answer evaluation
biobert_scorer_final = BERTScorer(
    model_type="dmis-lab/biobert-v1.1",
    num_layers=12,
    lang="en",
    rescale_with_baseline=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

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

def extract_final_answer(generated_text):
    if "</think>" in generated_text:
        final_answer = generated_text.split("</think>")[1].strip()
    elif "Final Answer:" in generated_text:
        final_answer = generated_text.split("Final Answer:")[1].strip()
    else:
        # Fallback: use the entire generated text if markers are missing
        final_answer = generated_text.split("### Response:")[1].strip()
    return final_answer

def evaluate_final_answer(pipeline, dataset, batch_size=16):
    all_final_predictions = []
    all_final_references = []
    
    # Prepare inputs (including chain-of-thought context if desired)
    formatted_inputs = [
        prompt_style.format(q,"") for q in dataset["questions"]
    ]
    
    # Batch generation and extraction
    for i in tqdm(range(0, len(formatted_inputs), batch_size), desc="Generating"):
        batch = formatted_inputs[i:i+batch_size]
        try:
            outputs = pipeline(
                batch,
                max_new_tokens=1200,
                temperature=0.1,
                do_sample=False,
                num_return_sequences=1,
            )
            for out in outputs:
                generated_text = out[0]["generated_text"]
                final_pred = extract_final_answer(generated_text)
                all_final_predictions.append(final_pred)
            all_final_references.extend(dataset["final_refs"][i:i+batch_size])
        except Exception as e:
            print(f"Generation failed for batch {i//batch_size}: {str(e)}")
            num_fail = len(batch)
            all_final_predictions.extend([""] * num_fail)
            all_final_references.extend(dataset["final_refs"][i:i+batch_size])
    
    # Compute BioBERTScore for final answer predictions
    P_final, R_final, F1_final = biobert_scorer_final.score(
        cands=all_final_predictions,
        refs=all_final_references,
        batch_size=256,
    )
    return F1_final.mean().item()

# Evaluate both models on final answers only
print("Evaluating Fine-tuned model...")
final_score_finetune = evaluate_final_answer(finetune_pipeline, test_data)
print(f"Fine-tuned Final Answer Accuracy: {final_score_finetune * 100:.2f}%")

print("Evaluating DeepSeek model...")
final_score_deepseek = evaluate_final_answer(deepseek_pipeline, test_data)
print(f"DeepSeek Final Answer Accuracy: {final_score_deepseek * 100:.2f}%")
