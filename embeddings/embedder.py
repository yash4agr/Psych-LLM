from sentence_transformers import SentenceTransformer
from typing import List

class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
    
    def embed_query(self, query: str) -> List[float]:
        return self.model.encode(query).tolist()