from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class ModelConfig:
    model_name: str = "meta-llama/Llama-3.2-3B" 
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    max_tokens: int = 2048
    temperature: float = 0.7

@dataclass
class VectorDBConfig:
    collection_name: str = "psychology_book"
    chunk_size: int = 500
    chunk_overlap: int = 50