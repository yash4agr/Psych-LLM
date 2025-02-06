import chromadb
from chromadb.config import Settings
from typing import List, Dict
from config import VectorDBConfig

class VectorStore:
    def __init__(self, config: VectorDBConfig):
        self.client = chromadb.Client(Settings(allow_reset=True))
        self.collection = self.client.create_collection(name=config.collection_name)
    
    def add_documents(self, chunks: List[tuple[str, Dict]], embeddings: List[List[float]]):
        texts = [chunk[0] for chunk in chunks]
        metadatas = [chunk[1] for chunk in chunks]
        ids = [f"doc_{i}" for i in range(len(chunks))]
        
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(self, query_embedding: List[float], n_results: int = 3) -> List[Dict]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results