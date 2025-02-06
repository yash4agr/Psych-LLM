from vllm import LLM, SamplingParams
from typing import List, Dict
from config import ModelConfig

class ResponseGenerator:
    def __init__(self, config: ModelConfig):
        self.llm = LLM(model=config.model_name)
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
    
    def generate_response(self, query: str, context: List[Dict]) -> str:
        prompt = self._create_prompt(query, context)
        response = self.llm.generate(prompt, self.sampling_params)
        return response[0].outputs[0].text

    def _create_prompt(self, query: str, context: List[Dict]) -> str:
        context_str = "\n".join([f"[{c['metadata']['section']} - {c['metadata']['subsection']}, Page {c['metadata']['page']}]: {c['text']}" for c in context])
        
        prompt = f"""Based on the following context from a psychology textbook, please answer the question. Include references to specific sections and page numbers in your response.

Context:
{context_str}

Question: {query}

Answer:"""
        return prompt