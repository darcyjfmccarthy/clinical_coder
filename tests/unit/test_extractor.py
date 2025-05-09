import pytest
from pathlib import Path
import os
from dotenv import load_dotenv
from src.agents.extractor import ExtractorAgent, ExtractionResult
from src.agents.preprocessor import PreprocessorAgent, OpenAIBackend

# Load environment variables
load_dotenv()

class MockPreprocessor:
    def process_note(self, raw_note: str):
        if not raw_note.strip():
            class EmptyNote:
                def __init__(self):
                    self.sections = {}
                    self.sentence_ids = {}
            return EmptyNote()
            
        class ProcessedNote:
            def __init__(self):
                self.sections = {
                    "Discharge Diagnosis": [
                        "Community-acquired pneumonia.",
                        "Essential (primary) hypertension.",
                        "Type 2 diabetes mellitus, uncomplicated."
                    ]
                }
                self.sentence_ids = {
                    "Discharge Diagnosis": ["diag_1", "diag_2", "diag_3"]
                }
        return ProcessedNote()

@pytest.fixture
def llm_backend():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable is required")
    return OpenAIBackend(api_key=api_key, model="deepseek-reasoner")

@pytest.fixture
def extractor(llm_backend):
    preprocessor = MockPreprocessor()
    return ExtractorAgent(preprocessor, llm_backend)

@pytest.fixture
def sample_note():
    fixture_path = Path(__file__).parent.parent / "fixtures" / "note_001.txt"
    return fixture_path.read_text()

def test_extractor_initialization(extractor):
    assert extractor is not None
    assert hasattr(extractor, 'pre')
    assert hasattr(extractor, 'llm')

def test_extract_diagnosis(extractor):
    result = extractor._extract_diagnosis(
        "Community-acquired pneumonia.",
        "diag_1"
    )
    assert isinstance(result, ExtractionResult)
    assert result.code.startswith("J")  # Should be a respiratory code
    assert result.title == "Community-acquired pneumonia."
    assert result.confidence == 0.8
    assert result.evidence == ["diag_1"]

def test_extract_diagnosis_invalid_input(extractor):
    result = extractor._extract_diagnosis("", "diag_1")
    assert result is None

def test_run_with_sample_note(extractor, sample_note):
    results = extractor.run(sample_note)
    
    assert len(results) == 3
    assert all(isinstance(r, dict) for r in results)
    
    # Verify first diagnosis (pneumonia)
    assert results[0]["code"].startswith("J")  # Should be a respiratory code
    assert results[0]["title"] == "Community-acquired pneumonia."
    assert results[0]["confidence"] == 0.8
    assert results[0]["evidence"] == ["diag_1"]
    
    # Verify second diagnosis (hypertension)
    assert results[1]["code"].startswith("I")  # Should be a circulatory code
    assert results[1]["title"] == "Essential (primary) hypertension."
    
    # Verify third diagnosis (diabetes)
    assert results[2]["code"].startswith("E")  # Should be an endocrine code
    assert results[2]["title"] == "Type 2 diabetes mellitus, uncomplicated."

def test_run_with_empty_note(extractor):
    results = extractor.run("")
    assert results == []

def test_run_with_missing_section(extractor):
    class EmptyPreprocessor:
        def process_note(self, raw_note: str):
            class ProcessedNote:
                def __init__(self):
                    self.sections = {}
                    self.sentence_ids = {}
            return ProcessedNote()
    
    extractor.pre = EmptyPreprocessor()
    results = extractor.run("some note")
    assert results == [] 