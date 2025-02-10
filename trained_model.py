from unsloth import FastLanguageModel

loaded_model, loaded_tokenizer = FastLanguageModel.from_pretrained(
    model_name = "trained_model",
    device_map = "auto",
    max_seq_length = 2048
)

