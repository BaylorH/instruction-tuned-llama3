from fastapi import FastAPI, Request, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import logging
import os

app = FastAPI()

# Paths
BASE_MODEL = "meta-llama/Llama-3.2-3B"
FINETUNED_MODEL_PATH = "lora_output"  # Directory where your finetuned weights live

# Determine which model to load
model_source = FINETUNED_MODEL_PATH if os.path.isdir(FINETUNED_MODEL_PATH) else BASE_MODEL

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_source)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Safety Alignment stub
def safety_alignment(query: str) -> bool:
    """
    TODO: implement actual safety checks
    """
    return True

# Few-shot prompting stub
def few_shot_prompt(query: str) -> str:
    """
    TODO: implement few-shot example retrieval
    """
    return ""

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    query = data.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Safety check
    if not safety_alignment(query):
        return {"response": "I’m sorry, but I can’t help with that."}

    # Build prompt with optional few-shot prefix
    prefix = few_shot_prompt(query)
    prompt = f"{prefix}Q: {query}\nA: Let's think step by step:"

    # Self-consistency sampling
    num_samples = 3
    generations = []
    start_time = time.time()
    for _ in range(num_samples):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            top_p=0.9
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generations.append(text)
    latency = time.time() - start_time
    logger.info(f"Query latency: {latency:.2f}s")

    # Majority vote on last line of each generation
    def extract_last_line(text: str) -> str:
        lines = text.strip().split("\n")
        return lines[-1] if lines else text

    answers = [extract_last_line(gen) for gen in generations]
    final_answer = max(set(answers), key=answers.count)

    return {
        "query": query,
        "final_answer": final_answer,
        "all_generations": generations,
        "latency_s": round(latency, 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
