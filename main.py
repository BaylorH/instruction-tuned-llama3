from fastapi import FastAPI, Request, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import logging

app = FastAPI()
MODEL_NAME = "meta-llama/Llama-3.2-3B"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
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

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    query = data.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Safety Alignment
    if not safety_alignment(query):
        return {"response": "I’m sorry, but I can’t help with that."}

    # Chain-of-thought prompt
    prompt = f"Q: {query}\nA: Let's think step by step:"

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

    # Majority vote on the last line of each generation
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
