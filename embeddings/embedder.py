from together import Together
from typing import List

class Embedder:
    def __init__(self, model_name: str):
        self.client = Together()
        self.model_name = model_name
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [
            item.embedding 
            for item in self.client.embeddings.create(
                model=self.model_name,
                input=texts,
            ).data
        ]
    
    def embed_query(self, query: str) -> List[float]:
        return self.client.embeddings.create(
                            model=self.model_name,
                            input=query,
                            ).data[0].embedding