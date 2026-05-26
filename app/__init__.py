"""Flask application factory."""
import os
from pathlib import Path
# pyrefly: ignore [missing-import]
from flask import Flask


def create_app() -> Flask:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent.parent / "templates"),
        static_folder=str(Path(__file__).parent.parent / "static"),
    )

    # ── Pastas necessárias ────────────────────────────────────────────────────
    base = Path(__file__).parent.parent
    for folder in ["static/uploads", "static/results", "models", "data"]:
        (base / folder).mkdir(parents=True, exist_ok=True)

    app.config["UPLOAD_FOLDER"] = str(base / "static" / "uploads")
    app.config["RESULT_FOLDER"] = str(base / "static" / "results")
    app.config["MODEL_PATH"]    = str(base / "models" / "best.pt")
    app.config["DB_PATH"]       = str(base / "data" / "germination.db")
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

    # ── Banco de dados ────────────────────────────────────────────────────────
    from app.database import init_db
    init_db(app.config["DB_PATH"])

    # ── Modelo de detecção ────────────────────────────────────────────────────
    from app.inference import load_model
    app.config["MODEL"] = load_model(app.config["MODEL_PATH"])

    # ── Rotas ─────────────────────────────────────────────────────────────────
    from app.routes import bp
    app.register_blueprint(bp)

    # ── Rotas WhatsApp ────────────────────────────────────────────────────────
    from app.whatsapp_routes import wp
    app.register_blueprint(wp)

    # Recovery de mensagens perdidas enquanto offline (best-effort)
    try:
        from app.whatsapp import get_client
        from app.processed_messages import get_store
        from app.recovery import recover_pending_messages
        from app.whatsapp_routes import process_webhook_message

        client = get_client()
        if client.is_configured():
            with app.app_context():
                recovered = recover_pending_messages(
                    client=client,
                    store=get_store(),
                    process_fn=process_webhook_message,
                )
            if recovered:
                print(f"[recovery] {recovered} mensagem(ns) recuperada(s) on startup")
    except Exception as exc:
        print(f"[recovery] erro ao tentar recuperar mensagens: {exc}")

    return app
