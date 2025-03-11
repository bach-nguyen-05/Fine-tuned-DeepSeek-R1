# DeepSeek Fine-Tune on Medical Dataset
For further understanding of the fine-tune and benchmark report, please visit [these slides](https://docs.google.com/presentation/d/1JePAXXyfgZA253JdJNgGeyQz2uqMjgwo7PQYNxCryz0/edit?usp=sharing)
## Fine-tune method
Model being used: DeepSeek-R1 Distill Llama3 8B <br>
The model has been fine-tuned using Low Rank Adaptation (LoRA), the code has been provided in finetune_80.py
+ HuggingFace Datasets (FreedomIntelligence/medical-o1-reasoning-SFT) was used for training and benchmarking
+ Supervised training: datasets include questions, chain of thoughts, and final answer (reasoning and final answers are used to calculate the loss value) 
+ 80% of the datasets (around 20000 rows) were used to fine-tune the model
+ 16 adapters are applied to 7 layers mentioned in the code 

Requirements for fine-tuning:
+ Unsloth (Optional) for fast loading model (AutoTokenizer and AutoCasualLm can be used instead).
+ datasets, PyTorch, and trl for training operations

## Benchmark Methods
5000 remaining rows of datasets were used for testing using the biobertscore metric
+ Benchmark prompt still triggers chain of thoughts before giving the answers (but not benchmarked)
+ Biobertscore is used mainly for verifying the quality of the final answers

## Result
The fine-tuned model yields more succinct chains of thoughts before giving more accurate answers. <br>
Because of limited testing resources, the 2 models are benchmarked on 2500 rows and 5000 rows (the training time for 5000 was long and might lead to some errors). <br>
![Diagram](![image](https://github.com/user-attachments/assets/c81c5998-c18b-4a35-a560-59adebec0e32))


