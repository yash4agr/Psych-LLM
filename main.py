from typing import Dict
import json
from utils.pdf_parser import PDFParser
from config import ModelConfig, VectorDBConfig
from embeddings.embedder import Embedder
from vector_store.chroma_store import VectorStore
from llm.llm_generator import ResponseGenerator
from utils.text_splitter import TextChunker

class RAGPipeline:
    def __init__(
        self,
        model_config: ModelConfig,
        vector_config: VectorDBConfig,
        sections_metadata: Dict
    ):
        self.embedder = Embedder(model_config.embedding_model)
        self.vector_store = VectorStore(vector_config)
        self.generator = ResponseGenerator(model_config)
        self.chunker = TextChunker(vector_config)
        self.pdf_parser = PDFParser()
        self.sections_metadata = sections_metadata
    
    def index_document(self, pdf_path: str):
        text = self.pdf_parser.parse_pdf(pdf_path)
        chunks = self.chunker.create_chunks_with_metadata(text, self.sections_metadata)
        embeddings = self.embedder.embed_documents([chunk[0] for chunk in chunks])
        self.vector_store.add_documents(chunks, embeddings)
    
    def query(self, query: str) -> str:
        query_embedding = self.embedder.embed_query(query)
        relevant_chunks = self.vector_store.query(query_embedding)
        response = self.generator.generate_response(query, relevant_chunks)
        return response

# Example usage
if __name__ == "__main__":
    # Load configurations
    model_config = ModelConfig()
    vector_config = VectorDBConfig()
    
    # Load sections metadata
    with open('Data/sections_metadata.json', 'r') as f:
        sections_metadata = json.load(f)
    
    # Initialize pipeline
    pipeline = RAGPipeline(model_config, vector_config, sections_metadata)
    
    # Index document
    pipeline.index_document("Data/book.pdf")
    
    # Query example
    query = "What are the main stages of sleep?"
    response = pipeline.query(query)
    print(response)