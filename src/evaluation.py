from datasets import load_dataset
from tqdm import tqdm
from sentence_transformers import util
from src.retrieval import get_dynamic_context
from src.pipeline import run_reflex_pipeline
from src.models import resources

def run_benchmark_validation(alpha, temp, n_samples=20):
    """Runs the pipeline against TruthfulQA."""
    print(f"\nLoading TruthfulQA Benchmark (n={n_samples})...")
    
    dataset = load_dataset("truthful_qa", "generation", split="validation", streaming=True)
    total_score = 0
    count = 0
    
    for row in tqdm(dataset.take(n_samples), total=n_samples, desc="Benchmarking"):
        question = row['question']
        gold_answer = row['best_answer']

        context = get_dynamic_context(question)
        result = run_reflex_pipeline(question, context, alpha, temp, n=3)
        
        emb_pred = resources.embedder.encode(result['answer'], convert_to_tensor=True)
        emb_gold = resources.embedder.encode(gold_answer, convert_to_tensor=True)
        score = float(util.pytorch_cos_sim(emb_pred, emb_gold).item())
        
        total_score += score
        count += 1
        
    return total_score / count

def grid_search_optimization():
    print("HYPERPARAMETER GRID SEARCH")
    alphas = [0.5, 0.7, 0.9]
    temps = [0.6, 0.8]
    
    results = []
    for temp in temps:
        for alpha in alphas:
            print(f"\nTesting: [Temp={temp} | Alpha={alpha}]")
            score = run_benchmark_validation(alpha, temp, n_samples=10)
            print(f"  -> TruthfulQA Score: {score:.4f}")
            results.append({"temp": temp, "alpha": alpha, "score": score})
            
    best = max(results, key=lambda x: x['score'])
    print(f"\nBest Config: Temp={best['temp']}, Alpha={best['alpha']}, Score={best['score']:.4f}")
    return best