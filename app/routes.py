"""Flask routes — API REST + serve da UI."""
from __future__ import annotations

import time as _time
import uuid
from pathlib import Path

# pyrefly: ignore [missing-import]
from flask import (
    Blueprint, Response, current_app, jsonify, render_template, request
)
# pyrefly: ignore [missing-import]
from werkzeug.utils import secure_filename

import sqlite3

from app.database import (
    delete_analysis, get_history, get_temporal_series, insert_analysis
)
from app.inference import run_inference

bp = Blueprint("main", __name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── UI ────────────────────────────────────────────────────────────────────────

@bp.route("/")
def index():
    return render_template("index.html", v=int(_time.time()))


@bp.route("/mcp")
def mcp_docs():
    return render_template("mcp.html", v=int(_time.time()))


@bp.route("/whatsapp")
def whatsapp():
    return render_template("whatsapp.html", v=int(_time.time()))


@bp.route("/favicon.ico")
def favicon():
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">🌱</text></svg>'
    return Response(svg.encode("utf-8"), mimetype="image/svg+xml")


# ── Status ────────────────────────────────────────────────────────────────────

@bp.route("/api/status")
def status():
    model = current_app.config.get("MODEL")
    model_path = current_app.config.get("MODEL_PATH")
    model_loaded = model is not None
    custom_model = Path(model_path).exists() if model_path else False

    return jsonify({
        "status":       "ok",
        "model_loaded": model_loaded,
        "custom_model": custom_model,
        "model_path":   model_path,
        "message":      "Modelo personalizado ativo ✅" if custom_model else "Usando YOLO11 pré-treinado (COCO) — adicione best.pt em models/ para resultados do seu dataset",
    })


# ── Análise de imagem ─────────────────────────────────────────────────────────

@bp.route("/api/analyze", methods=["POST"])
def analyze():
    # Valida arquivo
    if "image" not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada"}), 400

    file = request.files["image"]
    if file.filename == "" or not _allowed(file.filename):
        return jsonify({"error": "Tipo de arquivo inválido. Use PNG, JPG ou WEBP."}), 400

    day_label = request.form.get("day_label", "").strip() or None

    # Salva upload
    safe_name = secure_filename(file.filename)
    if not safe_name:  # se o nome original tinha só acentos, fica vazio
        safe_name = "imagem.jpg"
    filename  = f"{uuid.uuid4().hex[:12]}_{safe_name}"
    save_path = Path(current_app.config["UPLOAD_FOLDER"]) / filename
    try:
        file.save(str(save_path))
    except Exception as exc:
        return jsonify({"error": f"Erro ao salvar arquivo: {exc}"}), 500

    # Inferência
    try:
        result = run_inference(
            image_path=str(save_path),
            model=current_app.config["MODEL"],
            result_folder=current_app.config["RESULT_FOLDER"],
            conf_threshold=float(request.form.get("conf", 0.25)),
        )
    except Exception as exc:
        return jsonify({"error": f"Erro na inferência: {exc}"}), 500

    # Persiste no histórico
    try:
        record_id = insert_analysis(
            db_path=current_app.config["DB_PATH"],
            filename=file.filename,
            total_detected=result["total_detected"],
            germinated=result["germinated"],
            germination_rate=result["germination_rate"],
            leaf_avg=result["leaf_avg"],
            result_image=result["result_image"],
            day_label=day_label,
        )
    except Exception as exc:
        return jsonify({"error": f"Erro ao gravar no banco: {exc}"}), 500

    result["id"]        = record_id
    result["filename"]  = file.filename
    result["day_label"] = day_label
    result["total_folhas_estimadas"] = int(round(result["leaf_avg"] * result["germinated"]))
    return jsonify(result)


# ── Histórico ─────────────────────────────────────────────────────────────────

@bp.route("/api/history")
def history():
    limit = int(request.args.get("limit", 50))
    records = get_history(current_app.config["DB_PATH"], limit=limit)
    return jsonify(records)


@bp.route("/api/history/<int:analysis_id>", methods=["DELETE"])
def delete_record(analysis_id: int):
    ok = delete_analysis(current_app.config["DB_PATH"], analysis_id)
    if ok:
        return jsonify({"deleted": analysis_id})
    return jsonify({"error": "Registro não encontrado"}), 404


# ── Série temporal ────────────────────────────────────────────────────────────

@bp.route("/api/temporal")
def temporal():
    series = get_temporal_series(current_app.config["DB_PATH"])
    return jsonify(series)


# ── Estatísticas gerais ───────────────────────────────────────────────────────

@bp.route("/api/stats")
def stats():
    db = current_app.config["DB_PATH"]
    try:
        with sqlite3.connect(db) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("""
                SELECT
                    COUNT(*)                              AS total,
                    ROUND(AVG(germination_rate), 1)       AS avg_rate,
                    ROUND(MAX(germination_rate), 1)       AS best_rate,
                    ROUND(MIN(germination_rate), 1)       AS worst_rate,
                    ROUND(AVG(leaf_avg), 1)               AS avg_leaves,
                    SUM(total_detected)                   AS total_detected,
                    SUM(germinated)                       AS total_germinated,
                    COUNT(DISTINCT COALESCE(day_label, substr(timestamp,1,10))) AS days_tracked
                FROM analyses
            """).fetchone()
            # Distribuição por faixa de taxa
            buckets = conn.execute("""
                SELECT
                    CASE
                        WHEN germination_rate >= 80 THEN 'Ótima (≥80%)'
                        WHEN germination_rate >= 60 THEN 'Boa (60-79%)'
                        WHEN germination_rate >= 40 THEN 'Regular (40-59%)'
                        ELSE 'Baixa (<40%)'
                    END AS faixa,
                    COUNT(*) AS qtd
                FROM analyses GROUP BY faixa ORDER BY MIN(germination_rate) DESC
            """).fetchall()
        return jsonify({
            "summary": dict(row) if row else {},
            "distribution": [dict(b) for b in buckets],
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Chatbot ───────────────────────────────────────────────────────────────────

@bp.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    mensagem = (data.get("message") or "").strip()
    sender = (data.get("sender") or "").strip() or None
    if not mensagem:
        return jsonify({"error": "Mensagem vazia"}), 400
    from app.chatbot import gerar_resposta
    resposta = gerar_resposta(mensagem, current_app.config["DB_PATH"], sender=sender)
    return jsonify({"reply": resposta})
