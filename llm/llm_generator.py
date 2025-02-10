from together import Together
from typing import List, Dict, Any
from config import ModelConfig
import os

class ResponseGenerator:
    def __init__(self, config: ModelConfig):
        self.client = Together()
        self.model_name = config.model_name
    
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
        
        messages = self._create_prompt(query, context_items)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        return response.choices[0].message.content

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
        
        messages =  [
            {
            "role": "system", "content": f"""You are a knowledgeable psychology assistant. Use the provided sources to answer the question.
Context:
{context_str}

Instructions:

Use the provided context to ensure relevance in your response.
Maintain clarity, accuracy, and conciseness while avoiding unnecessary repetition.
Limit the response to 500 words.
Do not include closing phrases like "Best regards" or "Let me know if I can help you further."
Keep the response strictly relevant to the context.
Task:
Provide a clear, detailed, and well-structured answer to the following question, ensuring accuracy and relevance based on the context."""},
    {"role": "user", "content": f"""Question: {query}"""}
]
        return messages