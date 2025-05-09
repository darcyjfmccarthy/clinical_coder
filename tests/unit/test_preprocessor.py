import pathlib
import pytest

from src.agents.preprocessor import PreprocessorAgent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_DIR = pathlib.Path(__file__).parent.parent / "fixtures"
NOTE_PATH = FIXTURE_DIR / "note_001.txt"

@pytest.fixture(scope="module")
def sample_note() -> str:
    return NOTE_PATH.read_text(encoding="utf-8")

@pytest.fixture(scope="module")
def agent() -> PreprocessorAgent:
    # Use mock backend for tests to avoid API key requirements and model downloads
    return PreprocessorAgent(llm_backend="openai")

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_clean_text_removes_boilerplate(agent: PreprocessorAgent, sample_note: str):
    noisy = sample_note + "\n\nPage 2 of 7\n© 2025 Hospital XYZ\n"
    cleaned = agent._clean_text(noisy)
    assert "Page 2 of 7" not in cleaned
    assert "© 2025" not in cleaned
    assert "  " not in cleaned

def test_split_into_sections(agent: PreprocessorAgent, sample_note: str):
    secs = agent._split_into_sections(sample_note)
    # All expected keys present
    for key in ("Discharge Diagnosis", "Hospital Course", "Procedures", "Medications"):
        assert key in secs
    # Basic sanity on captured content
    assert "Community-acquired pneumonia" in secs["Discharge Diagnosis"]
    assert "amoxicillin" in secs["Medications"].lower()

def test_sentence_tokenisation(agent: PreprocessorAgent, sample_note: str):
    sentences = agent._tokenize_sentences(
        "BP controlled on amlodipine. Blood glucose 5-9 mmol/L on metformin."
    )
    assert sentences == [
        "BP controlled on amlodipine.",
        "Blood glucose 5-9 mmol/L on metformin.",
    ]

def test_process_note_end_to_end(agent: PreprocessorAgent, sample_note: str):
    note = agent.process_note(sample_note)
    # Sentence IDs contiguous and unique
    all_ids = [i for ids in note.sentence_ids.values() for i in ids]
    assert all_ids == list(range(len(all_ids)))
    # Principal section contains three sentences
    assert len(note.sections["Discharge Diagnosis"]) == 3
