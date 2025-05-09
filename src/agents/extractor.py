# src/agents/extractor.py
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class ExtractionResult:
    code: str
    title: str
    confidence: float
    evidence: List[str]
    raw_text: str

class ExtractorAgent:
    def __init__(self, preprocessor, llm_backend):
        self.pre = preprocessor
        self.llm = llm_backend

    def _calculate_confidence(self, code: str, title: str) -> float:
        """Calculate confidence score based on code and title match quality."""
        if not code or not title:
            return 0.0
        # Basic confidence scoring - can be enhanced with more sophisticated logic
        return 0.8 if code and title else 0.0

    def _extract_diagnosis(self, line: str, sentence_id: str) -> Optional[ExtractionResult]:
        """Extract a single diagnosis from a line of text."""
        try:
            code = self.llm.map_to_code(line)
            if not code:
                return None
                
            return ExtractionResult(
                code=code,
                title=line.strip(),
                confidence=self._calculate_confidence(code, line),
                evidence=[sentence_id],
                raw_text=line
            )
        except Exception as e:
            print(f"Error extracting diagnosis from line: {line}. Error: {str(e)}")
            return None

    def run(self, raw_note: str) -> List[Dict]:
        """
        Process a clinical note and extract coded diagnoses.
        
        Args:
            raw_note: Raw clinical note text
            
        Returns:
            List of dictionaries containing code, title, confidence, and evidence
        """
        try:
            note = self.pre.process_note(raw_note)
            diag_lines = note.sections.get("Discharge Diagnosis", [])
            sentence_ids = note.sentence_ids.get("Discharge Diagnosis", [])
            
            results = []
            for idx, (line, sentence_id) in enumerate(zip(diag_lines, sentence_ids)):
                extraction = self._extract_diagnosis(line, sentence_id)
                if extraction:
                    results.append({
                        "code": extraction.code,
                        "title": extraction.title,
                        "confidence": extraction.confidence,
                        "evidence": extraction.evidence,
                        "raw_text": extraction.raw_text
                    })
            
            return results
        except Exception as e:
            print(f"Error processing note: {str(e)}")
            return []
