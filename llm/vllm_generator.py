from vllm import LLM, SamplingParams
from typing import List, Dict, Any
from config import ModelConfig
import os

class ResponseGenerator:
    def __init__(self, config: ModelConfig):
        self.llm = LLM(model=config.model_name, dtype = 'half', device="auto")
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
    
    def generate_response(self, query: str, context: Dict[str, Any]) -> str:
        """
        Generate response based on query and retrieved context
        
        Args:
            query: User's query string
            context: Dictionary of query results from vector store
        
        Returns:
            Generated response as a string
        """
        # Extract documents and metadatas from context
        documents = context.get('documents', [[]])[0]
        metadatas = context.get('metadatas', [[]])[0]
        
        # Combine documents with their metadata
        context_items = [
            {
                'text': doc, 
                'metadata': metadata
            } 
            for doc, metadata in zip(documents, metadatas)
        ]
        
        prompt = self._create_prompt(query, context_items)
        
        response = self.llm.generate([prompt], self.sampling_params)
        return response[0].outputs[0].text

    def _create_prompt(self, query: str, context: List[Dict]) -> str:
        """
        Create a prompt with context and query
        
        Args:
            query: User's query string
            context: List of context dictionaries with 'text' and 'metadata'
        
        Returns:
            Formatted prompt string
        """
        context_str = "\n".join([
            f"[{c['metadata']['section']} - {c['metadata']['subsection']}, Page {c['metadata']['page']}]: {c['text']}" 
            for c in context
        ])
        
        prompt = f"""You are a knowledgeable psychology assistant. Use the provided sources to answer the question.
Context:
{context_str}

Instructions:

Use the provided context to ensure relevance in your response.
Maintain clarity, accuracy, and conciseness while avoiding unnecessary repetition.
Limit the response to 500 words.
Do not include closing phrases like "Best regards" or "Let me know if I can help you further."
Keep the response strictly relevant to the context.
Task:
Provide a clear, detailed, and well-structured answer to the following question, ensuring accuracy and relevance based on the context.

Question: {query}

Answer:"""
        return prompt