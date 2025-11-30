import torch

CONFIG = {
    # System Settings
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # Model Settings
    "llm_name": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "embed_name": "BAAI/bge-base-en-v1.5",
    "max_new_tokens": 256,
}