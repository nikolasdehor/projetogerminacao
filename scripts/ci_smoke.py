"""Smoke checks leves para CI.

Objetivo: validar caminho de código crítico sem iniciar a stack web inteira
(e sem exigir modelo YOLO baixado/carregado).
"""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)


def main() -> None:
    from app.database import init_db
    from app.guards import should_process_image
    from app.inference import _resolve_cell_count, parse_caption

    # Regras de parsing / helpers determinísticos.
    caption, capacity = parse_caption("Abertura 128")
    _assert(caption == "Abertura 128", "parse_caption retornou caption inesperada")
    _assert(capacity == 128, "parse_caption não extraíu capacidade esperada")

    resolved_cells, method = _resolve_cell_count(12, 12, None, raw_method="grid")
    _assert(resolved_cells == 12, "resolução de contagem de células falhou")
    _assert(method == "detected_visible", "método de contagem esperado mudou")

    should_process, reason = should_process_image(
        {
            "remoteJid": "5511999999999@s.whatsapp.net",
            "caption": "Teste de mensagem de DM",
        }
    )
    _assert(should_process is True, "DM com caption válida não foi processada")
    _assert(reason == "dm", f"razão inesperada ao validar DM: {reason}")

    # Checagem de inicialização de banco com path temporário (sem dependência de modelo).
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "germina-smoke.db"
        init_db(str(db_path))
        _assert(db_path.exists(), "Banco de dados de smoke não foi criado")


if __name__ == "__main__":
    main()
