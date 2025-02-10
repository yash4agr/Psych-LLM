from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free" 
    embedding_model: str = "togethercomputer/m2-bert-80M-2k-retrieval"
    max_tokens: int = 512
    temperature: float = 0.7

@dataclass
class VectorDBConfig:
    collection_name: str = "psychology_book"
    chunk_size: int = 500
    chunk_overlap: int = 50