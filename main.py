from fastapi import FastAPI, Request, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import logging
import os
import random
import json

app = FastAPI()

#model source
BASE_MODEL = "meta-llama/Llama-3.2-3B"
FINETUNED_MODEL_PATH = "lora_output"
model_source = FINETUNED_MODEL_PATH if os.path.isdir(FINETUNED_MODEL_PATH) else BASE_MODEL

#load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_source)
device = "cuda" if torch.cuda.is_available() else "cpu" #use gpu
model.to(device)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("dev_data.json") as f:
    dev_data = json.load(f)

#creates few shot prompts using random examples from dev_data
def few_shot_prompt() -> str:
    examples = random.sample(dev_data, 3)
    prompt = ""
    for ex in examples:
        cleaned_answer = ex["answer"].replace("####", "").strip() #removes "####"s
        prompt += f"Q: {ex['question']}\nA: Let's think step-by-step: {cleaned_answer}\n\n" #each few shot prompt will have this added to encourage chain of thought
    return prompt

def safety_alignment(query: str) -> bool:
    #TODO
    return True

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    query = data.get("query", "").strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if not safety_alignment(query):
        return {"response": "I'm sorry, but I can't help with that."} #simple safety response

    prefix = few_shot_prompt()
    prompt = f"{prefix}Q: {query}\nA:" #chain of thought should be learned by now from the few shot prompts

    num_samples = 5
    generations = []
    start_time = time.time()

    for _ in range(num_samples):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generations.append(text)

    latency = time.time() - start_time
    logger.info(f"Query latency: {latency:.2f}s")

#could use a method for only taking the end of the chain of thought
    def extract_last_line():
        pass  
