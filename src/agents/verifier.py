# src/agents/verifier.py
import sqlite3
from rapidfuzz import fuzz

class TerminologyVerifier:
    def __init__(self, db_path: str, trigram_threshold: float = 0.9):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.tri_threshold = trigram_threshold

    def verify(self, candidates):
        verified, invalid = [], []
        cur = self.conn.cursor()
        for cand in candidates:
            # First try exact code match
            cur.execute("SELECT title FROM icd10 WHERE code = ?", (cand["code"],))
            row = cur.fetchone()
            if row:
                cand["title_std"] = row[0]
                verified.append(cand)
                continue
            
            # Fallback: title fuzzy match using FTS5
            # Use proper FTS5 syntax with quoted search term
            search_term = f'"{cand["title"]}"'
            cur.execute("""
                SELECT code, title 
                FROM icd10 
                WHERE icd10 MATCH ? 
                ORDER BY rank 
                LIMIT 1
            """, (search_term,))
            row = cur.fetchone()
            
            if row and fuzz.token_sort_ratio(cand["title"], row[1]) / 100 >= self.tri_threshold:
                cand["code"] = row[0]
                cand["title_std"] = row[1]
                verified.append(cand)
            else:
                invalid.append(cand)
        return {"verified": verified, "invalid": invalid}
