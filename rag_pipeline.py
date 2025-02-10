from typing import Dict

from config import ModelConfig, VectorDBConfig
from utils.pdf_parser import PDFParser
from utils.text_splitter import TextChunker
from utils.timer import timer
from embeddings.embedder import Embedder
from vector_store.chroma_store import VectorStore
from llm.llm_generator import ResponseGenerator

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
        print("\nIndexing document...")

        with timer("PDF parsing"):
            text = self.pdf_parser.parse_pdf(pdf_path)

        with timer("Text chunking"):
            chunks = self.chunker.create_chunks_with_metadata(text, self.sections_metadata)
            print(f"Created {len(chunks)} chunks")

        with timer("Document embedding"):
            embeddings = self.embedder.embed_documents([chunk[0] for chunk in chunks])

        with timer("Vector store population"):
            self.vector_store.add_documents(chunks, embeddings)
    
    def query(self, query: str) -> str:
        query_embedding = self.embedder.embed_query(query)
        relevant_chunks = self.vector_store.query(query_embedding)
        response = self.generator.generate_response(query, relevant_chunks)
        return response
