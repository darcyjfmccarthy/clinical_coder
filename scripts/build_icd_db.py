# build_icd_db.py
"""One-off utility that converts an ICD-10-AM CSV dump into a compact
SQLite database with full-text and trigram search support.

Usage (bash):
    python build_icd_db.py --csv data/codes.csv \
                           --out data/icd10/icd10.sqlite

CSV schema (minimal):
    code,title,synonyms,major_block

    * code          - e.g. "J18.9"
    * title         - e.g. "Pneumonia, unspecified organism"
    * synonyms      - optional |-separated alt labels
    * major_block   - optional e.g. "Diseases of the respiratory system"

The script is idempotent: it removes any existing tables of the same name
before loading.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sqlite3
import sys
from typing import Iterable, Sequence

# ----------------------------------------------------------------------------
# SQL helpers
# ----------------------------------------------------------------------------

DDL_MAIN = """
DROP TABLE IF EXISTS icd10;
CREATE TABLE icd10 (
    code        TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    synonyms    TEXT DEFAULT '',
    major_block TEXT
);
"""

DDL_FTS = """
DROP TABLE IF EXISTS icd10_fts;
CREATE VIRTUAL TABLE icd10_fts USING fts5(
    code UNINDEXED,
    title,
    synonyms,
    content='icd10',
    tokenize='porter'
);
"""

DDL_TRIGRAM = """
DROP INDEX IF EXISTS icd10_trgm;
CREATE INDEX icd10_trgm ON icd10(title) USING trigram;
"""

INSERT_SQL = "INSERT INTO icd10 (code, title, synonyms, major_block) VALUES (?, ?, ?, ?);"

FTS_INSERT_SQL = "INSERT INTO icd10_fts(rowid, code, title, synonyms) VALUES (last_insert_rowid(), ?, ?, ?);"

# ----------------------------------------------------------------------------
# Core builders
# ----------------------------------------------------------------------------

def read_csv(path: pathlib.Path) -> Iterable[Sequence[str]]:
    """Yield rows from the ICD‑10 CSV file."""
    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        required_cols = {"code", "title"}
        if not required_cols.issubset(reader.fieldnames or {}):
            raise ValueError(
                f"CSV is missing required columns {required_cols} (found {reader.fieldnames})"
            )
        for row in reader:
            yield (
                row["code"].strip(),
                row["title"].strip(),
                row.get("synonyms", "").strip(),
                row.get("major_block", "").strip(),
            )


def init_db(db_path: pathlib.Path, enable_trigram: bool = True) -> sqlite3.Connection:
    """Create a fresh SQLite DB with schema and extensions."""
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=OFF;")

    conn.executescript(DDL_MAIN)
    conn.executescript(DDL_FTS)

    # Attempt to create trigram index (needs the "trigram" extension bundled with
    # SQLite 3.38+). If unavailable, silently continue.
    if enable_trigram:
        try:
            conn.enable_load_extension(True)
            # Load extension if packaged; otherwise rely on built‑in tokeniser.
            conn.load_extension("trigram")  # type: ignore[arg-type]
        except Exception:
            print("[WARN] Trigram extension unavailable; falling back to FTS only")
        else:
            conn.executescript(DDL_TRIGRAM)
            print("[INFO] Trigram index created")

    return conn


def populate_db(conn: sqlite3.Connection, rows: Iterable[Sequence[str]]) -> None:
    """Insert rows and populate FTS shadow table."""
    cursor = conn.cursor()
    for batch in batcher(rows, size=1000):
        cursor.executemany(INSERT_SQL, batch)
        cursor.executemany(FTS_INSERT_SQL, ((r[0], r[1], r[2]) for r in batch))
    conn.commit()


def batcher(iterable: Iterable[Sequence[str]], size: int):
    """Yield lists of *size* items from *iterable*."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def build_db(csv_path: pathlib.Path, db_path: pathlib.Path, *, skip_trigram: bool) -> None:
    rows = list(read_csv(csv_path))
    print(f"[INFO] Read {len(rows):,} rows from {csv_path}")

    conn = init_db(db_path, enable_trigram=not skip_trigram)
    populate_db(conn, rows)
    conn.close()
    print(f"[INFO] SQLite DB written to {db_path} ({db_path.stat().st_size/1024:.1f} KB)")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ICD‑10‑AM SQLite DB")
    p.add_argument("--csv", type=pathlib.Path, required=True, help="Path to source CSV file")
    p.add_argument("--out", type=pathlib.Path, required=True, help="Destination SQLite file")
    p.add_argument("--skip-trigram", action="store_true", help="Do not create trigram index")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    build_db(args.csv, args.out, skip_trigram=args.skip_trigram)


if __name__ == "__main__":
    main(sys.argv[1:])
