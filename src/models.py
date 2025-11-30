import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from src.config import CONFIG

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.embedder = None
        self._load_models()

    def _load_models(self):
        print(f"Loading Models on {CONFIG['device']}...")
        
        # 1. Load Embedder
        self.embedder = SentenceTransformer(CONFIG["embed_name"], device=CONFIG["device"])
        
        # 2. Load LLM (Quantized)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG["llm_name"])
        self.model = AutoModelForCausalLM.from_pretrained(
            CONFIG["llm_name"],
            quantization_config=bnb_config,
            device_map="auto"
        )
        print("Models Loaded Successfully.")

resources = ModelManager()