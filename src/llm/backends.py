from typing import List
from abc import ABC, abstractmethod
import openai
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    @abstractmethod
    def extract_conditions(self, text: str) -> List[str]:
        """Extract conditions from discharge diagnosis text."""
        pass

class HuggingFaceBackend(LLMBackend):
    """HuggingFace model backend."""
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    def extract_conditions(self, text: str) -> List[str]:
        prompt = f"""Extract the medical conditions from this discharge diagnosis text. 
        Return each condition as a separate line, preserving the numbering.
        Only return the conditions, nothing else.

        Text:
        {text}

        Conditions:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=200,
            num_return_sequences=1,
            temperature=0.1,
            do_sample=False
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        conditions_text = response.split("Conditions:")[-1].strip()
        return [line.strip() for line in conditions_text.split('\n') if line.strip()]

class OpenAIBackend(LLMBackend):
    """OpenAI model backend. Deepseek for now."""
    def __init__(self, api_key: str, model: str = "deepseek-reasoner"):
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model = model
    
    def extract_conditions(self, text: str) -> List[str]:
        prompt = f"""Extract the medical conditions from this discharge diagnosis text. 
        Return each condition as a separate line, preserving the numbering.
        Only return the conditions, nothing else.

        Text:
        {text}"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a medical text processing assistant. Extract conditions from discharge diagnoses."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        conditions_text = response.choices[0].message.content.strip()
        return [line.strip() for line in conditions_text.split('\n') if line.strip()]

    def map_to_code(self, text: str) -> str:
        """Map a medical diagnosis to its ICD-10 code."""
        if not text.strip():
            return ""
            
        prompt = f"""Given this medical diagnosis, return the most appropriate ICD-10 code.
        Only return the code, nothing else.

        Diagnosis: {text}"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a medical coding assistant. Return only ICD-10 codes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip() 