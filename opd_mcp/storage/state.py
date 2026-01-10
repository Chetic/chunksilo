"""Ingestion state management using SQLite."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class FileInfo:
    """Metadata about a file in the data source."""

    path: str
    hash: str
    last_modified: float


class IngestionState:
    """Manages the state of ingested files using a SQLite database."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY,
                    hash TEXT NOT NULL,
                    last_modified REAL NOT NULL,
                    doc_ids TEXT NOT NULL
                )
                """
            )

    def get_all_files(self) -> Dict[str, dict]:
        """Retrieve all tracked files and their metadata."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT path, hash, last_modified, doc_ids FROM files")
            return {
                row[0]: {
                    "hash": row[1],
                    "last_modified": row[2],
                    "doc_ids": row[3].split(",") if row[3] else [],
                }
                for row in cursor
            }

    def update_file_state(self, file_info: FileInfo, doc_ids: List[str]):
        """Update or insert the state for a file."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO files (path, hash, last_modified, doc_ids)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    hash=excluded.hash,
                    last_modified=excluded.last_modified,
                    doc_ids=excluded.doc_ids
                """,
                (
                    file_info.path,
                    file_info.hash,
                    file_info.last_modified,
                    ",".join(doc_ids),
                ),
            )

    def remove_file_state(self, path: str):
        """Remove a file from the state tracking."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM files WHERE path = ?", (path,))
