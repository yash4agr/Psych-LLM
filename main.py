import sys
import json
from typing import Optional

from config import ModelConfig, VectorDBConfig
from utils.timer import timer
from rag_pipeline import RAGPipeline

def run_pipeline(pdf_path: Optional[str] = None) -> None:
    """
    Run the RAG pipeline with the specified PDF or default path.
    
    Args:
        pdf_path: Optional path to the PDF file. Defaults to 'Data/book.pdf'
    """
    print("=== Psych-LLM ===")
    
    try:
        # Load configurations
        model_config = ModelConfig()
        vector_config = VectorDBConfig()
        
        # Load sections metadata
        with open('Data/sections_metadata.json', 'r') as f:
            sections_metadata = json.load(f)
    except FileNotFoundError:
        print("Error: sections_metadata.json not found in Data directory")
        return
    
    # Initialize pipeline
    pipeline = RAGPipeline(model_config, vector_config, sections_metadata)
    
    # Index document
    try:
        pipeline.index_document(pdf_path or "Data/book.pdf")
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path or 'Data/book.pdf'}")
        return
    
    # Interactive query loop
    print("\nEnter your questions (type 'exit' to quit)")
    while True:
        try:
            query = input("\nQuestion: ")
            if query.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break
            
            if not query.strip():
                continue
            
            print("\nGenerating response...")
            with timer("Response generation"):
                response = pipeline.query(query)
            
            print("\nAnswer:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Get PDF path from command line if provided
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(pdf_path)