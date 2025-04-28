import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# Get tokenizer
llm_tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-3B", 
    use_fast=True
)

if not llm_tokenizer.pad_token:
    llm_tokenizer.pad_token = llm_tokenizer.eos_token

# Get base model
llm_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B", 
    torch_dtype=torch.bfloat16
).to("cuda")

llm_model.resize_token_embeddings(len(llm_tokenizer))

# LoRA setup
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
)
adapted_model = get_peft_model(llm_model, lora_config)

# Load & sample dataset
instruction_dataset = load_dataset("MBZUAI/LaMini-instruction", split="train")
training_subset = instruction_dataset.shuffle(seed=42).select(range(100000))

# Tokenize & mask prompt in labels
def format_and_tokenize(batch):
    tokenized_inputs = []
    tokenized_masks = []
    tokenized_labels = []
    
    for instruction, answer in zip(batch["instruction"], batch["response"]):
        instruction_text = instruction.strip()
        response_text = answer.strip()
        
        formatted_prompt = f"{instruction_text}\n\n### Response:\n"
        complete_text = f"{formatted_prompt}{response_text}"
        
        # Tokenize complete text
        encoded_complete = llm_tokenizer(
            complete_text, 
            max_length=512,
            truncation=True
        )
        
        # Tokenize prompt only to get length
        encoded_prompt = llm_tokenizer(
            formatted_prompt,
            max_length=512,
            truncation=True
        )
        prompt_length = len(encoded_prompt["input_ids"])
        
        # Create labels with masking
        sequence_labels = encoded_complete["input_ids"].copy()
        for i in range(prompt_length):
            sequence_labels[i] = -100
            
        # Collect results
        tokenized_inputs.append(encoded_complete["input_ids"])
        tokenized_masks.append(encoded_complete["attention_mask"])
        tokenized_labels.append(sequence_labels)
    
    return {
        "input_ids": tokenized_inputs,
        "attention_mask": tokenized_masks,
        "labels": tokenized_labels
    }

# Process dataset with different variable names
processed_dataset = training_subset.map(
    format_and_tokenize,
    batched=True,
    remove_columns=training_subset.column_names
)

# Data collator for padding
data_collator = DataCollatorForSeq2Seq(
    llm_tokenizer,
    label_pad_token_id=-100,
    return_tensors="pt",
)

# Training arguments
config = TrainingArguments(
    output_dir="llama3_lora_finetuned",
    learning_rate=2e-4,
    num_train_epochs=2,
    fp16=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    warmup_steps=200,
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    report_to="none"
)

# Trainer
trainer = Trainer(
    data_collator=data_collator,
    train_dataset=processed_dataset,
    model=adapted_model,
    args=config
)

# Train
if __name__ == "__main__":
    trainer.train()
    trainer.save_model()
