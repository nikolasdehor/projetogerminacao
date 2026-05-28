"""Testes unitários para app/guards.py."""
from __future__ import annotations

import os
import pytest


# ---------------------------------------------------------------------------
# Helpers para limpar o cache do lru_cache entre testes
# ---------------------------------------------------------------------------

def _clear_caches():
    from app.guards import _load_whitelist, _load_keywords
    _load_whitelist.cache_clear()
    _load_keywords.cache_clear()


# ---------------------------------------------------------------------------
# should_process_image
# ---------------------------------------------------------------------------

class TestShouldProcessImage:
    def setup_method(self):
        _clear_caches()

    def test_broadcast_skip(self, monkeypatch):
        monkeypatch.setattr("app.guards._load_whitelist", lambda: frozenset())
        monkeypatch.setattr("app.guards._load_keywords", lambda: ("germina",))
        msg = {"remoteJid": "5562999999999@broadcast", "caption": "germina"}
        process, reason = _call(msg)
        assert process is False
        assert reason == "broadcast"

    def test_dm_passes(self, monkeypatch):
        monkeypatch.setattr("app.guards._load_whitelist", lambda: frozenset())
        monkeypatch.setattr("app.guards._load_keywords", lambda: ("germina",))
        msg = {"remoteJid": "5562999999999@s.whatsapp.net", "caption": "foto aleatória"}
        process, reason = _call(msg)
        assert process is True
        assert reason == "dm"

    def test_group_in_whitelist_passes(self, monkeypatch):
        whitelist = frozenset(["5562111111111@g.us"])
        monkeypatch.setattr("app.guards._load_whitelist", lambda: whitelist)
        monkeypatch.setattr("app.guards._load_keywords", lambda: ("germina",))
        msg = {"remoteJid": "5562111111111@g.us", "caption": ""}
        process, reason = _call(msg)
        assert process is True
        assert reason == "whitelist"

    def test_group_keyword_passes(self, monkeypatch):
        monkeypatch.setattr("app.guards._load_whitelist", lambda: frozenset())
        monkeypatch.setattr("app.guards._load_keywords", lambda: ("germina", "bandeja"))
        msg = {"remoteJid": "5562222222222@g.us", "caption": "foto da bandeja hoje"}
        process, reason = _call(msg)
        assert process is True
        assert reason == "keyword"

    def test_group_keyword_accent_insensitive(self, monkeypatch):
        monkeypatch.setattr("app.guards._load_whitelist", lambda: frozenset())
        monkeypatch.setattr("app.guards._load_keywords", lambda: ("germinação",))
        msg = {"remoteJid": "5562333333333@g.us", "caption": "GERMINACAO hoje"}
        process, reason = _call(msg)
        assert process is True
        assert reason == "keyword"

    def test_group_no_keyword_silent(self, monkeypatch):
        monkeypatch.setattr("app.guards._load_whitelist", lambda: frozenset())
        monkeypatch.setattr("app.guards._load_keywords", lambda: ("germina", "bandeja"))
        msg = {"remoteJid": "5562444444444@g.us", "caption": "olha que cachorro lindo"}
        process, reason = _call(msg)
        assert process is False
        assert reason == "no_keyword"

    def test_group_no_caption_silent(self, monkeypatch):
        monkeypatch.setattr("app.guards._load_whitelist", lambda: frozenset())
        monkeypatch.setattr("app.guards._load_keywords", lambda: ("germina",))
        msg = {"remoteJid": "5562555555555@g.us", "caption": ""}
        process, reason = _call(msg)
        assert process is False
        assert reason == "no_keyword"

    def test_should_process_image_unknown_jid_silent(self):
        msg = {"remoteJid": "xxx@newsletter", "caption": "qualquer coisa"}
        process, reason = _call(msg)
        assert process is False
        assert reason == "unknown_jid_format"


# ---------------------------------------------------------------------------
# passes_post_inference_guard
# ---------------------------------------------------------------------------

class TestPassesPostInferenceGuard:
    def setup_method(self):
        os.environ.pop("GUARD_MIN_DETECTIONS", None)
        os.environ.pop("GUARD_MIN_MEAN_CONF", None)

    def test_low_count_silences(self):
        detections = [{"confidence": 0.9}, {"confidence": 0.9}]
        mean_conf = 0.9
        passes, reason = _call_post(detections, mean_conf)
        assert passes is False
        assert reason == "low_count"

    def test_low_conf_silences(self):
        detections = [
            {"confidence": 0.3},
            {"confidence": 0.3},
            {"confidence": 0.3},
        ]
        mean_conf = 0.3
        passes, reason = _call_post(detections, mean_conf)
        assert passes is False
        assert reason == "low_conf"

    def test_ok_processes(self):
        detections = [
            {"confidence": 0.8},
            {"confidence": 0.7},
            {"confidence": 0.9},
        ]
        mean_conf = 0.8
        passes, reason = _call_post(detections, mean_conf)
        assert passes is True
        assert reason == "ok"

    def test_custom_env_vars(self, monkeypatch):
        monkeypatch.setenv("GUARD_MIN_DETECTIONS", "5")
        monkeypatch.setenv("GUARD_MIN_MEAN_CONF", "0.9")
        detections = [{"confidence": 0.95}] * 4
        mean_conf = 0.95
        passes, reason = _call_post(detections, mean_conf)
        assert passes is False
        assert reason == "low_count"


# ---------------------------------------------------------------------------
# Wrappers que importam dentro do teste para isolar o módulo
# ---------------------------------------------------------------------------

def _call(msg):
    from app.guards import should_process_image
    return should_process_image(msg)


def _call_post(detections, mean_conf):
    from app.guards import passes_post_inference_guard
    return passes_post_inference_guard(detections, mean_conf)
