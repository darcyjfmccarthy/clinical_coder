import pytest
import sqlite3
import tempfile
from src.agents.verifier import TerminologyVerifier

@pytest.fixture
def test_db():
    # Create a temporary database for testing
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    # Create and populate the test database
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Create the icd10 table with FTS5
    cur.execute("""
        CREATE VIRTUAL TABLE icd10 USING fts5(
            code UNINDEXED,
            title
        )
    """)
    
    # Insert some test data
    test_data = [
        ("A0100", "Typhoid fever, unspecified"),
        ("B020", "Zoster encephalitis"),
        ("C50911", "Malignant neoplasm of unsp site of right female breast")
    ]
    cur.executemany("INSERT INTO icd10 (code, title) VALUES (?, ?)", test_data)
    conn.commit()
    conn.close()
    
    yield db_path
    
    # Cleanup - ensure connection is closed before deletion
    try:
        conn = sqlite3.connect(db_path)
        conn.close()
    except:
        pass
    
    import os
    try:
        os.unlink(db_path)
    except:
        pass

def test_verify_exact_match(test_db):
    verifier = TerminologyVerifier(test_db)
    candidates = [
        {"code": "A0100", "title": "Typhoid fever, unspecified"},
        {"code": "B020", "title": "Zoster encephalitis"}
    ]
    
    result = verifier.verify(candidates)
    
    assert len(result["verified"]) == 2
    assert len(result["invalid"]) == 0
    assert result["verified"][0]["code"] == "A0100"
    assert result["verified"][0]["title_std"] == "Typhoid fever, unspecified"
    assert result["verified"][1]["code"] == "B020"
    assert result["verified"][1]["title_std"] == "Zoster encephalitis"

def test_verify_fuzzy_match(test_db):
    verifier = TerminologyVerifier(test_db, trigram_threshold=0.8)
    candidates = [
        {"code": "X99.9", "title": "Typhoid fever, unspecified"},  # Wrong code but correct title
        {"code": "Y88.8", "title": "Zoster encephalitis"}   # Wrong code but correct title
    ]
    
    result = verifier.verify(candidates)
    
    assert len(result["verified"]) == 2
    assert len(result["invalid"]) == 0
    assert result["verified"][0]["code"] == "A0100"  # Should be updated to correct code
    assert result["verified"][1]["code"] == "B020"  # Should be updated to correct code

def test_verify_invalid_candidates(test_db):
    verifier = TerminologyVerifier(test_db)
    candidates = [
        {"code": "X99.9", "title": "Unknown disease"},  # Both code and title are invalid
        {"code": "Y88.8", "title": "Non-existent condition"}
    ]
    
    result = verifier.verify(candidates)
    
    assert len(result["verified"]) == 0
    assert len(result["invalid"]) == 2
    assert result["invalid"][0]["code"] == "X99.9"
    assert result["invalid"][1]["code"] == "Y88.8"

def test_verify_mixed_candidates(test_db):
    verifier = TerminologyVerifier(test_db)
    candidates = [
        {"code": "A0100", "title": "Typhoid fever, unspecified"},  # Exact match
        {"code": "X99.9", "title": "Typhoid fever, unspecified"},  # Fuzzy match
        {"code": "Y88.8", "title": "Unknown disease"} # Invalid
    ]
    
    result = verifier.verify(candidates)
    
    assert len(result["verified"]) == 2
    assert len(result["invalid"]) == 1
    assert result["verified"][0]["code"] == "A0100"
    assert result["verified"][1]["code"] == "A0100"  # Should be updated to correct code
    assert result["invalid"][0]["code"] == "Y88.8" 