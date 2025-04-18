import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1) Configuration
BASE_MODEL = "meta-llama/Llama-3.2-3B"
OUTPUT_DIR = "lora_output"
DATA_PATH   = "instructions.jsonl"  # JSONL: {"instruction","input","output"} per line

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2) Load base model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model = prepare_model_for_kbit_training(model)

# 3) PEFT/LoRA config
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, peft_config)

# 4) Load your instruction dataset
class InstructionDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer):
        self.examples = []
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                prompt = f"### Instruction:\n{obj['instruction']}\n### Input:\n{obj['input']}\n### Response:\n"
                target = obj["output"]
                enc = tokenizer(prompt, truncation=True, padding="longest", max_length=512)
                tgt = tokenizer(target, truncation=True, padding="longest", max_length=512)
                enc["labels"] = tgt["input_ids"]
                self.examples.append(enc)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return {k: torch.tensor(v) for k, v in self.examples[i].items()}

dataset = InstructionDataset(DATA_PATH, tokenizer)

# 5) TrainingArguments & Trainer
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=50,
    save_steps=200,
    learning_rate=1e-4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# 6) Run training and save
trainer.train()
trainer.save_model(OUTPUT_DIR)
