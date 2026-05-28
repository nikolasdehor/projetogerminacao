from __future__ import annotations

import json
import logging
import os
import unicodedata
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_CONFIG_BASE = Path(__file__).resolve().parent.parent / "config"


def _normalize(text: str) -> str:
    """Lowercase + remove acentos (unicodedata NFD)."""
    nfd = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


@lru_cache(maxsize=1)
def _load_whitelist() -> frozenset[str]:
    """Carrega config/group_whitelist.json. Tolera ausência (retorna frozenset vazio)."""
    path = _CONFIG_BASE / "group_whitelist.json"
    if not path.exists():
        return frozenset()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return frozenset(g["jid"] for g in data.get("groups", []) if "jid" in g)
    except Exception as exc:
        logger.warning("guard: falha ao carregar whitelist: %s", exc)
        return frozenset()


@lru_cache(maxsize=1)
def _load_keywords() -> tuple[str, ...]:
    """Carrega config/germina_keywords.json. Defaults se ausente."""
    _defaults = ("germina", "germinação", "bandeja", "plaqueta", "semente", "sementes", "contar")
    path = _CONFIG_BASE / "germina_keywords.json"
    if not path.exists():
        return _defaults
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        kws = tuple(str(k) for k in data.get("keywords", []) if k)
        return kws if kws else _defaults
    except Exception as exc:
        logger.warning("guard: falha ao carregar keywords: %s", exc)
        return _defaults


def _jid_truncated(jid: str) -> str:
    """Retorna sufixo seguro do JID para log (últimos 6 chars antes do @)."""
    local = jid.split("@")[0] if "@" in jid else jid
    return f"...{local[-6:]}@{jid.split('@')[1]}" if "@" in jid and len(local) > 6 else jid


def should_process_image(message: dict) -> tuple[bool, str]:
    """
    Decide se imagem deve ser processada antes da inferência.

    Returns (process, reason).
    reason in {'dm', 'whitelist', 'keyword', 'broadcast', 'no_keyword', 'no_whitelist', 'unknown_jid_format'}
    """
    remote_jid: str = message.get("remoteJid", "")

    if remote_jid.endswith("@broadcast"):
        return False, "broadcast"

    if remote_jid.endswith("@s.whatsapp.net"):
        return True, "dm"

    if not remote_jid.endswith("@g.us"):
        return False, "unknown_jid_format"

    group_jid = remote_jid
    whitelist = _load_whitelist()
    if group_jid in whitelist:
        return True, "whitelist"

    caption: str = message.get("caption", "") or ""
    caption_norm = _normalize(caption)
    keywords = _load_keywords()
    for kw in keywords:
        if _normalize(kw) in caption_norm:
            return True, "keyword"

    jid_short = _jid_truncated(group_jid)
    logger.info(
        "guard_skip group=%s reason=no_keyword caption_snippet=%.40s",
        jid_short,
        caption,
    )
    return False, "no_keyword"


def passes_post_inference_guard(
    detections: list[dict], mean_conf: float
) -> tuple[bool, str]:
    """
    Decide se resposta deve ser enviada após inferência.

    Returns (pass, reason).
    reason in {'ok', 'low_count', 'low_conf'}
    """
    min_detections = int(os.getenv("GUARD_MIN_DETECTIONS", "3"))
    min_conf = float(os.getenv("GUARD_MIN_MEAN_CONF", "0.55"))

    if len(detections) < min_detections:
        return False, "low_count"

    if mean_conf < min_conf:
        return False, "low_conf"

    return True, "ok"
