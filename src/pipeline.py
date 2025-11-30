import torch
import re
from typing import Dict
from src.config import CONFIG
from src.models import resources
from src.utils import format_prompt
from src.retrieval import get_dynamic_context
from sentence_transformers import util

def generate_text(prompt: str, temp: float) -> str:
    """Generates text using the loaded LLM."""
    inputs = resources.tokenizer(prompt, return_tensors="pt").to(CONFIG["device"])
    
    with torch.no_grad():
        outputs = resources.model.generate(
            **inputs,
            max_new_tokens=CONFIG["max_new_tokens"],
            temperature=temp,
            do_sample=True, # Required for voting variety
            pad_token_id=resources.tokenizer.eos_token_id,
            eos_token_id=resources.tokenizer.eos_token_id
        )
    
    text = resources.tokenizer.decode(outputs[0], skip_special_tokens=False)
    if "<|start_header_id|>assistant<|end_header_id|>" in text:
        return text.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1].replace("<|eot_id|>", "").strip()
    return text

def get_metrics(question: str, answer: str, context: str) -> Dict[str, float]:
    """Calculates Similarity (Fact-Check) and Confidence (Self-Check)."""
    
    # 1. Semantic Similarity
    emb_a = resources.embedder.encode(answer, convert_to_tensor=True)
    emb_b = resources.embedder.encode(context, convert_to_tensor=True)
    sim_score = float(util.pytorch_cos_sim(emb_a, emb_b).item())
    
    # 2. Self-Confidence
    eval_prompt = format_prompt(
        "Rate confidence from 0.0 to 1.0.", 
        f"Question: {question}\nAnswer: {answer}\nFormat: 'Score: 0.9'"
    )
    eval_output = generate_text(eval_prompt, temp=0.1) 
    match = re.search(r"Score:\s*([0-1](?:\.\d+)?)", eval_output)
    conf_score = float(match.group(1)) if match else 0.5
    
    return {"sim": sim_score, "conf": conf_score}

def run_reflex_pipeline(question: str, context: str, alpha: float, temp: float, n: int = 3) -> Dict:
    """Main Best-of-N Voting Pipeline."""
    candidates = []
    
    for i in range(n):
        # Generate
        prompt = format_prompt(
            "You are a helpful assistant. Use context if relevant.",
            f"Context: {context}\n\nQuestion: {question}"
        )
        answer = generate_text(prompt, temp)
        
        # Evaluate
        metrics = get_metrics(question, answer, context)
        final_score = (metrics['sim'] * alpha) + (metrics['conf'] * (1 - alpha))
        
        candidates.append({
            "answer": answer,
            "score": final_score,
            "raw_metrics": metrics
        })

    return max(candidates, key=lambda x: x['score'])