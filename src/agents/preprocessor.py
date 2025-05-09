from typing import Dict, List, Tuple, Optional, Literal
import re
from dataclasses import dataclass
import nltk
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import openai
import os
from dotenv import load_dotenv
from abc import ABC, abstractmethod

# Load environment variables from .env file
load_dotenv()

@dataclass
class ProcessedNote:
    """Container for processed clinical note data."""
    sections: Dict[str, List[str]]  # section_name -> list of sentences
    sentence_ids: Dict[str, List[int]]  # section_name -> list of sentence IDs
    raw_text: str  # original text

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

class PreprocessorAgent:
    """Agent responsible for preprocessing clinical notes for coding tasks."""
    
    def __init__(self, 
                 llm_backend: Literal["huggingface", "openai"] = "openai",
                 model_name: Optional[str] = "deepseek-ai/deepseek-coder-1.3b-base",
                 openai_api_key: Optional[str] = None,
                 openai_model: str = "deepseek-reasoner"):
        """Initialize the preprocessor with the specified LLM backend."""
        if llm_backend == "huggingface":
            if not model_name:
                raise ValueError("model_name is required for huggingface backend")
            self.llm = HuggingFaceBackend(model_name)
        elif llm_backend == "openai":
            # Try to get API key from environment if not provided
            api_key = openai_api_key or os.getenv("DEEPSEEK_API_KEY")
            print(f"Debug: DEEPSEEK_API_KEY exists in env: {'DEEPSEEK_API_KEY' in os.environ}")
            print(f"Debug: DEEPSEEK_API_KEY value: {os.getenv('DEEPSEEK_API_KEY')}")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY environment variable or openai_api_key parameter is required for openai backend")
            self.llm = OpenAIBackend(api_key, openai_model)
        else:
            raise ValueError(f"Unsupported LLM backend: {llm_backend}")
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def _clean_text(self, text: str) -> str:
        """Remove boilerplate and OCR artifacts."""
        # Remove common OCR artifacts
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
        
        # Remove common boilerplate
        boilerplate_patterns = [
            r'CONFIDENTIAL.*?DOCUMENT',
            r'Page \d+ of \d+',
            r'Generated on:.*?\n',
            r'©.*?\n',
        ]
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Normalize whitespace but preserve newlines
        lines = []
        for line in text.split('\n'):
            # Clean each line
            line = re.sub(r'\s+', ' ', line)  # Normalize whitespace within line
            line = re.sub(r'[^\w\s.,;:!?()\-]', '', line)  # Keep relevant punctuation and hyphens
            if line.strip():  # Only keep non-empty lines
                lines.append(line.strip())
        
        return '\n'.join(lines)
    
    def _split_into_sections(self, text: str) -> Dict[str, str]:
        """Split text into canonical sections."""
        sections = {
            'Discharge Diagnosis': '',
            'Hospital Course': '',
            'Procedures': '',
            'Medications': ''
        }
        
        # Common section headers and their variations
        section_patterns = {
            'Discharge Diagnosis': r'(?:DISCHARGE|Discharge|Final) (?:DIAGNOSIS|Diagnosis|Diagnoses)(?::|\n)(.*?)(?=\n\s*(?:Hospital Course|Procedures|Medications|$))',
            'Hospital Course': r'Hospital Course(?::|\n)(.*?)(?=\n\s*(?:Procedures|Medications|Discharge Diagnosis|$))',
            'Procedures': r'Procedures(?::|\n)(.*?)(?=\n\s*(?:Medications|Discharge Diagnosis|Hospital Course|$))',
            'Medications': r'Medications(?::|\n)(.*?)(?=\n\s*(?:Discharge Diagnosis|Hospital Course|Procedures|$))'
        }
        
        print("Input text:", repr(text))  # Debug print
        
        for section, pattern in section_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                # Get the matched text and preserve newlines
                section_text = match.group(1).strip()
                
                # For Discharge Diagnosis, ensure numbered items stay together
                if section == 'Discharge Diagnosis':
                    # Split by newlines but keep numbered items together
                    lines = section_text.split('\n')
                    processed_lines = []
                    current_item = None
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                            
                        # Check if line starts with a number followed by a period
                        if re.match(r'^\d+\.', line):
                            if current_item:
                                processed_lines.append(current_item)
                            current_item = line
                        else:
                            if current_item:
                                current_item += ' ' + line
                            else:
                                processed_lines.append(line)
                    
                    if current_item:
                        processed_lines.append(current_item)
                    
                    section_text = '\n'.join(processed_lines)
                
                sections[section] = section_text
                print(f"Found {section}:", repr(sections[section]))  # Debug print
            else:
                print(f"No match for {section}")  # Debug print
        
        return sections
    
    def _tokenize_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK."""
        # First, handle numbered lists by combining numbers with their content
        lines = text.split('\n')
        processed_lines = []
        current_item = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with a number followed by a period
            if re.match(r'^\d+\.', line):
                if current_item:
                    processed_lines.append(current_item)
                # Start new item with the number and period
                current_item = line
            else:
                if current_item:
                    # Append to current item if it exists
                    current_item += ' ' + line
                else:
                    processed_lines.append(line)
        
        if current_item:
            processed_lines.append(current_item)
        
        # Join the processed lines
        text = ' '.join(processed_lines)
        
        # Use NLTK for sentence tokenization
        sentences = nltk.sent_tokenize(text)
        
        # Post-process to ensure numbered items stay together
        final_sentences = []
        for sent in sentences:
            # If sentence starts with a number and period, it's a complete numbered item
            if re.match(r'^\d+\.', sent):
                final_sentences.append(sent)
            # Otherwise, it's a regular sentence
            else:
                final_sentences.append(sent)
        
        return final_sentences
    
    def _extract_conditions(self, text: str) -> List[str]:
        """Extract conditions from discharge diagnosis text using the configured LLM backend."""
        return self.llm.extract_conditions(text)

    def process_note(self, text: str) -> ProcessedNote:
        """Process a clinical note through the preprocessing pipeline."""
        # Clean the text
        cleaned_text = self._clean_text(text)
        print("Cleaned text:", repr(cleaned_text))  # Debug print
        
        # Split into sections
        sections = self._split_into_sections(cleaned_text)
        
        # Process each section
        processed_sections = {}
        sentence_ids = {}
        current_id = 0
        
        for section_name, section_text in sections.items():
            if section_text:
                if section_name == "Discharge Diagnosis":
                    # Use LLM to extract conditions
                    sentences = self._extract_conditions(section_text)
                else:
                    sentences = self._tokenize_sentences(section_text)
                    
                processed_sections[section_name] = sentences
                sentence_ids[section_name] = list(range(current_id, current_id + len(sentences)))
                current_id += len(sentences)
                print(f"Processed {section_name}:", repr(sentences))  # Debug print
            else:
                processed_sections[section_name] = []
                sentence_ids[section_name] = []
        
        return ProcessedNote(
            sections=processed_sections,
            sentence_ids=sentence_ids,
            raw_text=text
        )
