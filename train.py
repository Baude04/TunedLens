from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator

from src.tuned_lens_model import SmolLMTunedLens
from src.training import train_tuned_lens

# Load the base model and tokenizer
model_id = "HuggingFaceTB/SmolLM2-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto"
)
print("Model and tokenizer loaded.")

tuned_lens = SmolLMTunedLens(model).to(model.device, dtype=torch.bfloat16)

# Load FineWeb-Edu (Taking ~60,000 articles for optimal Lens training)
print("Downloading FineWeb-Edu subset...")
raw_dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu", 
    data_files="sample/10BT/000_00000.parquet", 
    split="train[:10000]"
)

# Keep only the text column to avoid tokenization issues with metadata
raw_dataset = raw_dataset.select_columns(["text"])

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_dataset = raw_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing"
)

# Chunk into strict 128-token blocks to save memory
block_size = 128    
def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    return {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
lm_dataset = tokenized_dataset.map(
    group_texts,
    batched=True,
    desc=f"Grouping into blocks of {block_size}"
)

# Create DataLoader
train_dataloader = DataLoader(
    lm_dataset,
    batch_size=1,
    collate_fn=default_data_collator,
    shuffle=True
)

# Execute Training
print("Starting training on FineWeb-Edu...")
train_tuned_lens(
    tuned_lens=tuned_lens,
    base_model=model,
    dataloader=train_dataloader,
    epochs=1,
    device=model.device,
    saving_rate=1000
)