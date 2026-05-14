"""Embeddings semânticos via sentence-transformers para busca contextual e RAG."""
from __future__ import annotations

import hashlib
import os
import sqlite3
import textwrap

import numpy as np

_encoder = None


def get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        # CPU explícito para não brigar com MPS do treino YOLO
        _encoder = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2",
            device="cpu",
        )
    return _encoder


def encode(text: str) -> bytes:
    model = get_encoder()
    vec = model.encode(
        text or "",
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    return vec.tobytes()


def decode(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # Vetores já normalizados pelo encode; dot product = cosine similarity
    return float(np.dot(a, b))


# ── Busca em chat_history ──────────────────────────────────────────────────────

def find_similar_messages(
    db_path: str,
    sender: str,
    query: str,
    top_k: int = 5,
    exclude_recent: int = 20,
    threshold: float = 0.25,
) -> list[dict]:
    """Busca mensagens semanticamente similares à query, excluindo as mais recentes."""
    try:
        query_vec = decode(encode(query))
    except Exception:
        return []

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """SELECT id, role, content, embedding
               FROM chat_history
               WHERE sender = ? AND embedding IS NOT NULL
               ORDER BY created_at DESC
               LIMIT 500""",
            (sender,),
        ).fetchall()

    if not rows:
        return []

    candidates = rows[exclude_recent:]
    if not candidates:
        return []

    scored: list[tuple[float, dict]] = []
    for row_id, role, content, blob in candidates:
        try:
            vec = decode(blob)
            score = cosine_sim(query_vec, vec)
            if score > threshold:
                scored.append((score, {"id": row_id, "role": role, "content": content, "score": score}))
        except Exception:
            continue

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]


# ── Knowledge base (RAG agronômico) ───────────────────────────────────────────

KNOWLEDGE_BASE_SQL = """
CREATE TABLE IF NOT EXISTS knowledge_base (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    source     TEXT    NOT NULL,
    chunk      TEXT    NOT NULL,
    chunk_hash TEXT    NOT NULL UNIQUE,
    embedding  BLOB
);
"""


def _chunk_text(text: str, max_chars: int = 500, overlap: int = 50) -> list[str]:
    """Divide texto em chunks por parágrafo, juntando até max_chars."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= max_chars:
            current = (current + " " + para).strip() if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) > max_chars:
                for piece in textwrap.wrap(para, max_chars, break_long_words=False):
                    chunks.append(piece)
                current = chunks[-1][-overlap:] if chunks else ""
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks


def ingest_knowledge_base(db_path: str) -> int:
    """Ingere conhecimento agronômico na tabela knowledge_base. Idempotente via hash."""
    from app.chatbot import DICAS_GERMINACAO  # import local para evitar circular

    with sqlite3.connect(db_path) as conn:
        conn.execute(KNOWLEDGE_BASE_SQL)
        conn.commit()

    inserted = 0

    # Fonte 1: arquivo MD do sistema
    md_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "Sistema de Visão Computacional para Monitoramento de Germinação e Crescimento de Mudas (1).md",
    )
    if os.path.exists(md_path):
        with open(md_path, encoding="utf-8") as f:
            md_text = f.read()
        if "## References" in md_text:
            md_text = md_text.split("## References")[0]
        chunks = _chunk_text(md_text, max_chars=500, overlap=50)
        for chunk in chunks:
            inserted += _upsert_chunk(db_path, source="sistema_md", chunk=chunk)

    # Fonte 2: DICAS_GERMINACAO
    for dica in DICAS_GERMINACAO:
        inserted += _upsert_chunk(db_path, source="dicas", chunk=dica)

    return inserted


def _upsert_chunk(db_path: str, source: str, chunk: str) -> int:
    """Insere chunk se não existir (dedupe por hash). Retorna 1 se inseriu, 0 se já existia."""
    chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
    try:
        embedding = encode(chunk)
    except Exception:
        embedding = None

    with sqlite3.connect(db_path) as conn:
        existing = conn.execute(
            "SELECT id FROM knowledge_base WHERE chunk_hash = ?", (chunk_hash,)
        ).fetchone()
        if existing:
            return 0
        conn.execute(
            "INSERT INTO knowledge_base (source, chunk, chunk_hash, embedding) VALUES (?, ?, ?, ?)",
            (source, chunk, chunk_hash, embedding),
        )
        conn.commit()
    return 1


def find_relevant_knowledge(
    db_path: str,
    query: str,
    top_k: int = 3,
    threshold: float = 0.3,
) -> list[dict]:
    """Busca chunks do knowledge_base mais relevantes para a query."""
    try:
        query_vec = decode(encode(query))
    except Exception:
        return []

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT chunk, embedding FROM knowledge_base WHERE embedding IS NOT NULL"
        ).fetchall()

    if not rows:
        return []

    scored: list[tuple[float, str]] = []
    for chunk, blob in rows:
        try:
            vec = decode(blob)
            score = cosine_sim(query_vec, vec)
            if score > threshold:
                scored.append((score, chunk))
        except Exception:
            continue

    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"chunk": chunk, "score": score} for score, chunk in scored[:top_k]]


# ── Análises históricas ────────────────────────────────────────────────────────

def find_similar_analyses(
    db_path: str,
    query: str,
    top_k: int = 2,
    threshold: float = 0.25,
) -> list[dict]:
    """Busca análises do histórico semanticamente similares à query."""
    try:
        query_vec = decode(encode(query))
    except Exception:
        return []

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """SELECT analysis_summary, analysis_embedding
               FROM analyses
               WHERE analysis_embedding IS NOT NULL
               ORDER BY id DESC
               LIMIT 100"""
        ).fetchall()

    if not rows:
        return []

    scored: list[tuple[float, str]] = []
    for summary, blob in rows:
        try:
            vec = decode(blob)
            score = cosine_sim(query_vec, vec)
            if score > threshold:
                scored.append((score, summary))
        except Exception:
            continue

    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"summary": s, "score": sc} for sc, s in scored[:top_k]]


# ── Backfill helpers ───────────────────────────────────────────────────────────

def backfill_embeddings(db_path: str) -> int:
    """Re-embeda mensagens de chat_history que estão com embedding NULL."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, content FROM chat_history WHERE embedding IS NULL LIMIT 1000"
        ).fetchall()
    count = 0
    for row_id, content in rows:
        try:
            blob = encode(content)
            with sqlite3.connect(db_path) as conn:
                conn.execute("UPDATE chat_history SET embedding=? WHERE id=?", (blob, row_id))
                conn.commit()
            count += 1
        except Exception:
            continue
    return count


def backfill_analyses(db_path: str) -> int:
    """Gera summary_embedding para analyses antigas sem embedding."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """SELECT id, timestamp, total_detected, germinated, germination_rate,
                      leaf_avg, analysis_summary
               FROM analyses
               WHERE analysis_embedding IS NULL
               ORDER BY id DESC LIMIT 500"""
        ).fetchall()
    count = 0
    for row in rows:
        row_id, ts, total, germinated, rate, leaf, summary = row
        if not summary:
            avaliacao = "excelente" if rate >= 80 else ("boa" if rate >= 60 else ("regular" if rate >= 40 else "baixa"))
            summary = (
                f"Análise em {ts}: bandeja com {total} mudas detectadas, "
                f"{germinated} germinadas ({rate:.1f}%), "
                f"média {leaf:.1f} folhas/planta. Avaliação: {avaliacao}."
            )
        try:
            blob = encode(summary)
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "UPDATE analyses SET analysis_summary=?, analysis_embedding=? WHERE id=?",
                    (summary, blob, row_id),
                )
                conn.commit()
            count += 1
        except Exception:
            continue
    return count
