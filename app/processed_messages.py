"""Persistencia de mensagens processadas para dedupe entre webhook + recovery."""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional


class ProcessedMessageStore:
    """SQLite minimalista pra rastrear quais message IDs ja foram processados."""

    def __init__(self, db_path: str | Path = "data/processed_messages.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_messages (
                    message_id TEXT PRIMARY KEY,
                    processed_at INTEGER NOT NULL,
                    message_timestamp INTEGER
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_processed_at ON processed_messages(processed_at)"
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), timeout=10.0)

    def is_processed(self, message_id: str) -> bool:
        if not message_id:
            return False
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "SELECT 1 FROM processed_messages WHERE message_id = ?",
                (message_id,),
            )
            return cur.fetchone() is not None

    def mark_processed(
        self,
        message_id: str,
        message_timestamp: Optional[int] = None,
    ) -> None:
        if not message_id:
            return
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO processed_messages
                    (message_id, processed_at, message_timestamp)
                VALUES (?, ?, ?)
                """,
                (message_id, int(time.time()), message_timestamp),
            )
            conn.commit()

    def latest_timestamp(self) -> Optional[int]:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "SELECT MAX(message_timestamp) FROM processed_messages"
            )
            row = cur.fetchone()
            return int(row[0]) if row and row[0] is not None else None


_store: Optional[ProcessedMessageStore] = None


def get_store() -> ProcessedMessageStore:
    global _store
    if _store is None:
        _store = ProcessedMessageStore()
    return _store
