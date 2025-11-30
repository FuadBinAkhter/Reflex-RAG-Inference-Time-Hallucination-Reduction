from src.evaluation import grid_search_optimization
from src.pipeline import run_reflex_pipeline
from src.retrieval import get_dynamic_context

if __name__ == "__main__":
    print("--- Phase 1: Optimization ---")
    best_params = grid_search_optimization()
    
    print("\n--- Phase 2: Production Demo ---")
    user_q = "Why is the sky blue?"
    context = get_dynamic_context(user_q)
    
    final_result = run_reflex_pipeline(
        user_q, 
        context, 
        alpha=best_params['alpha'], 
        temp=best_params['temp'], 
        n=10
    )
    
    print(f"\nQuestion: {user_q}")
    print(f"Optimized Answer: {final_result['answer']}")
    print(f"Confidence Score: {final_result['score']:.3f}")