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


def init_db(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(CREATE_SQL)
        conn.commit()


def insert_analysis(
    db_path: str,
    filename: str,
    total_detected: int,
    germinated: int,
    germination_rate: float,
    leaf_avg: float,
    result_image: str,
    day_label: Optional[str] = None,
) -> int:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO analyses
               (timestamp, filename, total_detected, germinated, germination_rate, leaf_avg, result_image, day_label)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts, filename, total_detected, germinated, germination_rate, leaf_avg, result_image, day_label),
        )
        conn.commit()
        return cur.lastrowid


def get_history(db_path: str, limit: int = 50) -> list[dict]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def delete_analysis(db_path: str, analysis_id: int) -> bool:
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        conn.commit()
        return cur.rowcount > 0


def get_temporal_series(db_path: str) -> list[dict]:
    """Retorna série temporal agrupada por day_label ou dia de timestamp."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT
                COALESCE(day_label, substr(timestamp, 1, 10)) AS day,
                AVG(germination_rate)  AS avg_germination_rate,
                AVG(leaf_avg)          AS avg_leaf_count,
                COUNT(*)               AS num_analyses
               FROM analyses
               GROUP BY day
               ORDER BY day ASC"""
        ).fetchall()
    return [dict(r) for r in rows]
