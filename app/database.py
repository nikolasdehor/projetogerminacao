"""SQLite helper — histórico de análises."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Optional


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS analyses (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT    NOT NULL,
    filename         TEXT    NOT NULL,
    total_detected   INTEGER NOT NULL DEFAULT 0,
    germinated       INTEGER NOT NULL DEFAULT 0,
    germination_rate REAL    NOT NULL DEFAULT 0.0,
    leaf_avg         REAL    NOT NULL DEFAULT 0.0,
    result_image     TEXT,
    day_label        TEXT
);
"""

CHAT_HISTORY_SQL = """
CREATE TABLE IF NOT EXISTS chat_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    sender     TEXT    NOT NULL,
    role       TEXT    NOT NULL,
    content    TEXT    NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_chat_history_sender_date
    ON chat_history(sender, created_at DESC);
"""


def init_db(db_path: str) -> None:
    from app.embeddings import KNOWLEDGE_BASE_SQL
    with sqlite3.connect(db_path) as conn:
        conn.execute(CREATE_SQL)
        for stmt in CHAT_HISTORY_SQL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)
        conn.execute(KNOWLEDGE_BASE_SQL)
        # Colunas idempotentes
        for alter in (
            "ALTER TABLE chat_history ADD COLUMN embedding BLOB",
            "ALTER TABLE analyses ADD COLUMN analysis_summary TEXT",
            "ALTER TABLE analyses ADD COLUMN analysis_embedding BLOB",
            "ALTER TABLE analyses ADD COLUMN source TEXT",
            "ALTER TABLE analyses ADD COLUMN sender TEXT",
            "ALTER TABLE analyses ADD COLUMN caption TEXT",
        ):
            try:
                conn.execute(alter)
            except sqlite3.OperationalError:
                pass
        conn.commit()


def insert_chat_message(db_path: str, sender: str, role: str, content: str) -> None:
    embedding: Optional[bytes] = None
    try:
        from app.embeddings import encode as _encode
        embedding = _encode(content)
    except Exception:
        pass

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO chat_history (sender, role, content, embedding) VALUES (?, ?, ?, ?)",
            (sender, role, content, embedding),
        )
        conn.commit()


def get_chat_history(db_path: str, sender: str, limit: int = 20) -> list[dict]:
    """Retorna últimas N mensagens do sender em ordem cronológica."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """SELECT role, content FROM chat_history
               WHERE sender = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (sender, limit),
        ).fetchall()
    # Reverte para ordem cronológica (mais antiga primeiro)
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]


def insert_analysis(
    db_path: str,
    filename: str,
    total_detected: int,
    germinated: int,
    germination_rate: float,
    leaf_avg: float,
    result_image: str,
    day_label: Optional[str] = None,
    source: str = "web",
    sender: Optional[str] = None,
    caption: Optional[str] = None,
) -> int:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if germination_rate >= 80:
        avaliacao = "excelente"
    elif germination_rate >= 60:
        avaliacao = "boa"
    elif germination_rate >= 40:
        avaliacao = "regular"
    else:
        avaliacao = "baixa"

    summary = (
        f"Análise em {ts}: bandeja com {total_detected} mudas detectadas, "
        f"{germinated} germinadas ({germination_rate:.1f}%), "
        f"média {leaf_avg:.1f} folhas/planta. Avaliação: {avaliacao}."
    )

    analysis_embedding: Optional[bytes] = None
    try:
        from app.embeddings import encode as _encode
        analysis_embedding = _encode(summary)
    except Exception:
        pass

    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO analyses
               (timestamp, filename, total_detected, germinated, germination_rate,
                leaf_avg, result_image, day_label, analysis_summary, analysis_embedding, source, sender, caption)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts, filename, total_detected, germinated, germination_rate,
             leaf_avg, result_image, day_label, summary, analysis_embedding, source, sender, caption),
        )
        conn.commit()
        return cur.lastrowid


def get_history(db_path: str, limit: int = 20, offset: int = 0) -> list[dict]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT id, timestamp, filename, total_detected, germinated, germination_rate,
                      leaf_avg, result_image, day_label, analysis_summary, caption,
                      CASE WHEN filename LIKE 'whatsapp_%' THEN 'whatsapp'
                           WHEN sender IS NOT NULL AND sender != '' THEN 'whatsapp'
                           WHEN source IS NOT NULL THEN source
                           ELSE 'web' END AS source,
                      CASE WHEN sender IS NOT NULL AND sender != '' THEN sender
                           WHEN filename LIKE 'whatsapp_%' THEN SUBSTR(filename, 10)
                           ELSE NULL END AS sender
               FROM analyses ORDER BY id DESC LIMIT ? OFFSET ?""",
            (limit, offset)
        ).fetchall()
    return [dict(r) for r in rows]


def count_analyses(db_path: str) -> int:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()
    return row[0] if row else 0


def delete_analysis(db_path: str, analysis_id: int) -> bool:
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        conn.commit()
        return cur.rowcount > 0


def get_temporal_series(db_path: str) -> list[dict]:
    """Retorna série temporal agrupada por data (YYYY-MM-DD)."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT
                substr(timestamp, 1, 10)  AS day,
                AVG(germination_rate)     AS avg_germination_rate,
                AVG(leaf_avg)             AS avg_leaf_count,
                COUNT(*)                  AS num_analyses
               FROM analyses
               GROUP BY substr(timestamp, 1, 10)
               ORDER BY day ASC"""
        ).fetchall()
    return [dict(r) for r in rows]
