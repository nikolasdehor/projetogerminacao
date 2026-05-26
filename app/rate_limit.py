"""Rate limit por usuario para mensagens em grupos."""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional


class RateLimitStore:
    """SQLite simples para rastrear quantas msgs/imagens cada usuario mandou."""

    def __init__(self, db_path: str | Path = "data/rate_limits.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    sender TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    ts INTEGER NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sender_kind_ts ON events(sender, kind, ts)")
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), timeout=10.0)

    def record(self, sender: str, kind: str) -> None:
        if not sender or not kind:
            return
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO events (sender, kind, ts) VALUES (?, ?, ?)",
                (sender, kind, int(time.time())),
            )
            conn.commit()

    def count_in_window(self, sender: str, kind: str, window_seconds: int = 3600) -> int:
        if not sender or not kind:
            return 0
        cutoff = int(time.time()) - window_seconds
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "SELECT COUNT(*) FROM events WHERE sender = ? AND kind = ? AND ts >= ?",
                (sender, kind, cutoff),
            )
            return int(cur.fetchone()[0])

    def prune_older_than(self, days: int = 7) -> None:
        cutoff = int(time.time()) - days * 86400
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM events WHERE ts < ?", (cutoff,))
            conn.commit()


_store: Optional[RateLimitStore] = None


def get_rate_limit_store() -> RateLimitStore:
    global _store
    if _store is None:
        _store = RateLimitStore()
    return _store
