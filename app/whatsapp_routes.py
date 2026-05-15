"""Rotas do WhatsApp — webhook + painel de configuração."""
from __future__ import annotations

import base64
import json
import os
import time as _time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from flask import (
    Blueprint, current_app, jsonify, render_template, request
)

# Dedup de message_id para descartar reentregas da Evolution API
_seen_msg_ids: deque[str] = deque(maxlen=500)
_seen_ids_lock = Lock()

# Pool de workers com backpressure: max 2 inferências YOLO simultâneas
_MAX_WORKERS = 2
_MAX_QUEUE = 20
_executor = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="wa-worker")
_queue_lock = Lock()
_queue_pending = {"count": 0}


def _process_async(app, payload: dict) -> None:
    """Processa mensagem em background com app context."""
    try:
        with app.app_context():
            _handle_message(payload)
    except Exception as exc:
        app.logger.exception(f"[worker] erro processando mensagem: {exc}")
    finally:
        with _queue_lock:
            _queue_pending["count"] = max(0, _queue_pending["count"] - 1)

from app.whatsapp import get_client
from app.inference import parse_caption as _parse_caption

wp = Blueprint("whatsapp", __name__)


def _extract_quoted_text(msg: dict) -> str | None:
    """Retorna texto da mensagem citada (reply/quote do WhatsApp), ou None."""
    ctx = msg.get("extendedTextMessage", {}).get("contextInfo", {})
    if not ctx:
        return None
    quoted = ctx.get("quotedMessage", {})
    if not quoted:
        return None
    return (
        quoted.get("conversation")
        or quoted.get("extendedTextMessage", {}).get("text")
        or quoted.get("imageMessage", {}).get("caption")
        or quoted.get("documentMessage", {}).get("caption")
        or "(mídia sem legenda)"
    )


# ── Página de configuração ────────────────────────────────────────────────────

@wp.route("/whatsapp")
def whatsapp_config():
    """Página de configuração do WhatsApp."""
    return render_template("whatsapp.html", v=int(_time.time()))


# ── API: Status da conexão ────────────────────────────────────────────────────

@wp.route("/api/whatsapp/status")
def whatsapp_status():
    """Retorna status da conexão WhatsApp."""
    client = get_client()
    if not client.is_configured():
        return jsonify({
            "configured": False,
            "state": "unconfigured",
            "message": "Evolution API não configurada. Preencha as variáveis no .env",
        })

    try:
        status = client.get_instance_status()
        state = status.get("instance", {}).get("state", status.get("state", "close"))
        return jsonify({
            "configured": True,
            "state": state,
            "message": _state_message(state),
        })
    except RuntimeError as e:
        return jsonify({
            "configured": True,
            "state": "error",
            "message": f"Erro ao conectar na Evolution API: {e}",
        })


def _state_message(state: str) -> str:
    msgs = {
        "open": "✅ WhatsApp conectado e pronto!",
        "close": "🔴 WhatsApp desconectado",
        "connecting": "🟡 Conectando...",
    }
    return msgs.get(state, f"Estado: {state}")


# ── API: Criar instância ──────────────────────────────────────────────────────

@wp.route("/api/whatsapp/connect", methods=["POST"])
def whatsapp_connect():
    """Cria instância e retorna QR Code."""
    client = get_client()
    if not client.is_configured():
        return jsonify({"error": "Evolution API não configurada"}), 400

    data = request.get_json(silent=True) or {}
    webhook_url = data.get("webhook_url", "").strip().rstrip("/")

    if not webhook_url:
        return jsonify({"error": "Informe a webhook_url (URL pública do seu servidor)"}), 400

    try:
        # Cria instância
        result = client.create_instance(webhook_url=webhook_url + "/api/whatsapp/webhook")
        qrcode_data = result.get("qrcode", {})

        # Se não veio QR no create, tenta buscar
        if not qrcode_data:
            qr_result = client.get_qrcode()
            qrcode_data = qr_result

        return jsonify({
            "success": True,
            "qrcode": qrcode_data.get("base64", ""),
            "pairingCode": qrcode_data.get("pairingCode", ""),
            "message": "Escaneie o QR Code com seu WhatsApp",
        })

    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500


# ── API: Desconectar ──────────────────────────────────────────────────────────

@wp.route("/api/whatsapp/disconnect", methods=["POST"])
def whatsapp_disconnect():
    """Desconecta o WhatsApp."""
    client = get_client()
    try:
        client.logout_instance()
        return jsonify({"success": True, "message": "WhatsApp desconectado"})
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500


# ── API: QR Code atualizado ───────────────────────────────────────────────────

@wp.route("/api/whatsapp/qrcode")
def whatsapp_qrcode():
    """Retorna QR Code atualizado para conexão."""
    client = get_client()
    try:
        result = client.get_qrcode()
        return jsonify({
            "base64": result.get("base64", ""),
            "pairingCode": result.get("pairingCode", ""),
        })
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500


# ── Health check ──────────────────────────────────────────────────────────────

@wp.route("/api/whatsapp/health", methods=["GET"])
def whatsapp_health():
    with _queue_lock:
        pending = _queue_pending["count"]
    return jsonify({
        "status": "ok",
        "queue_pending": pending,
        "queue_max": _MAX_QUEUE,
        "workers": _MAX_WORKERS,
        "seen_ids_cache": len(_seen_msg_ids),
    }), 200


# ── Webhook: Recebe mensagens do WhatsApp ─────────────────────────────────────

@wp.route("/api/whatsapp/webhook", methods=["POST"])
def whatsapp_webhook():
    """
    Recebe eventos da Evolution API.
    - MESSAGES_UPSERT: mensagens novas (texto ou imagem)
    - CONNECTION_UPDATE: status da conexão
    """
    payload = request.get_json(silent=True) or {}
    event = payload.get("event", "").upper()

    print(f"📱 Webhook recebido: {event}")

    if event in ("MESSAGES.UPSERT", "MESSAGES_UPSERT"):
        with _queue_lock:
            if _queue_pending["count"] >= _MAX_QUEUE:
                current_app.logger.warning(
                    f"[backpressure] fila cheia ({_queue_pending['count']}/{_MAX_QUEUE}), descartando webhook"
                )
                return jsonify({"received": True, "queued": False, "reason": "queue_full"}), 200
            _queue_pending["count"] += 1

        app = current_app._get_current_object()
        _executor.submit(_process_async, app, payload)

    elif event in ("CONNECTION.UPDATE", "CONNECTION_UPDATE"):
        state = payload.get("data", {}).get("state", "unknown")
        current_app.logger.info(f"[whatsapp] conexão: {state}")

    # Retorna 200 imediatamente para a Evolution API não reenviar por timeout
    return jsonify({"received": True, "queued": True}), 200


# ── Handler de mensagens ──────────────────────────────────────────────────────

def _handle_message(payload: dict):
    """Processa uma mensagem recebida pelo WhatsApp."""
    from flask import current_app

    data = payload.get("data", {})
    key = data.get("key", {})

    # Ignora mensagens enviadas por nós mesmos
    if key.get("fromMe", False):
        return

    # Dedup thread-safe: descarta reentregas do mesmo webhook pela Evolution API
    msg_id = key.get("id", "")
    if msg_id:
        with _seen_ids_lock:
            if msg_id in _seen_msg_ids:
                current_app.logger.info(f"[dedup] message_id duplicado ignorado: {msg_id}")
                return
            _seen_msg_ids.append(msg_id)

    remote_jid = key.get("remoteJid", "")
    # Extrai número limpo (5511999999999)
    sender = remote_jid.split("@")[0] if "@" in remote_jid else remote_jid

    if not sender:
        return

    msg = data.get("message", {})

    # Filtra tipos de mensagem que não devem ser processados (reactions, protocol, etc.)
    _ALLOWED_MSG_TYPES = {"imageMessage", "conversation", "extendedTextMessage", "documentMessage"}
    if msg and not any(k in msg for k in _ALLOWED_MSG_TYPES):
        current_app.logger.info(f"[skip] tipo de mensagem ignorado: {list(msg.keys())}")
        return
    client = get_client()

    # ── Imagem recebida → roda inferência ──────────────────────────────────
    if "imageMessage" in msg or "documentMessage" in msg:
        raw_caption = (
            msg.get("imageMessage", {}).get("caption")
            or msg.get("documentMessage", {}).get("caption")
            or ""
        ).strip() or None
        _handle_image_message(client, sender, payload, raw_caption=raw_caption)
        return

    # ── Texto recebido → chatbot ou comandos ───────────────────────────────
    text = (
        msg.get("conversation")
        or msg.get("extendedTextMessage", {}).get("text")
        or ""
    ).strip().lower()

    if not text:
        return

    quoted = _extract_quoted_text(msg)

    # Comandos especiais
    if text in ("status", "estatísticas", "estatisticas", "stats"):
        _handle_status_command(client, sender)
    elif text in ("dica", "dicas", "tip"):
        _handle_tip_command(client, sender)
    elif text in ("histórico", "historico", "history"):
        _handle_history_command(client, sender)
    elif text in ("ajuda", "help", "menu", "?"):
        _handle_help_command(client, sender)
    else:
        # Passa para o GerminaBot (IA) com contexto de quote se houver
        _handle_chat_message(client, sender, text, quoted=quoted)


def _handle_image_message(client, sender: str, payload: dict, raw_caption: str | None = None):
    """Processa imagem: baixa, roda YOLO, envia resultado."""
    from flask import current_app
    from app.inference import run_inference
    from app.database import insert_analysis

    upload_dir = current_app.config["UPLOAD_FOLDER"]
    result_dir = current_app.config["RESULT_FOLDER"]

    caption, tray_capacity_override = _parse_caption(raw_caption)

    # Notifica que estamos processando (com indicador "digitando..." para feedback visual)
    client.send_presence(sender, "composing", delay_ms=3000)
    client.send_text(sender, "🔬 *Analisando sua imagem...*\nAguarde um momento!")

    # Baixa a imagem
    image_path = client.download_media(payload, upload_dir)
    if not image_path:
        client.send_text(sender, "⚠️ Não consegui baixar a imagem. Tente enviar novamente.")
        return

    # Roda inferência (capacidade da caption passada para sanity check interno)
    try:
        result = run_inference(
            image_path=image_path,
            model=current_app.config["MODEL"],
            result_folder=result_dir,
            conf_threshold=0.5,
            tray_capacity_override=tray_capacity_override,
        )
    except Exception as e:
        client.send_text(sender, f"❌ Erro na análise: {e}")
        return

    capacity = result["cells_detected"]
    germinated = result["germinated"]
    rate = result["germination_rate"]

    # Salva no banco
    try:
        insert_analysis(
            db_path=current_app.config["DB_PATH"],
            filename=f"whatsapp_{sender}",
            total_detected=result["total_detected"],
            germinated=germinated,
            germination_rate=rate,
            leaf_avg=result["leaf_avg"],
            result_image=result["result_image"],
            day_label=caption,
            source="whatsapp",
            sender=sender,
            caption=caption,
        )
    except Exception:
        pass  # Não impede a resposta
    if rate >= 75:
        emoji = "🟢"
        avaliacao = "Excelente germinação! Bandeja pronta para o próximo ciclo."
    elif rate >= 55:
        emoji = "🟡"
        avaliacao = "Germinação moderada. Considere replantio nos espaços vazios ou verifique uniformidade."
    else:
        emoji = "🔴"
        avaliacao = "Atenção! Taxa baixa. Avalie qualidade das sementes, substrato, umidade e temperatura."

    label_linha = f"🏷️ *Tratamento:* {caption}\n" if caption else ""
    cells_origin = result.get("cells_origin", "detected")
    origem_label = {"caption": "informada", "detected": "detectada", "fallback_default": "estimada"}.get(cells_origin, "detectada")
    cells_warning = result.get("cells_warning")
    warning_linha = f"\n{cells_warning}" if cells_warning else ""

    texto = (
        f"🌱 *Análise da Bandeja — GerminaVision*\n"
        f"{label_linha}\n"
        f"📊 *Resultados:*\n"
        f"• Plantas germinadas: {germinated} de {capacity} células ({rate}%) [{origem_label}]\n"
        f"• Folhas por planta (média): {result['leaf_avg']}\n"
        f"• Total de folhas estimadas: {int(round(result['leaf_avg'] * germinated))}\n"
        f"• Tempo de análise: {result['inference_time_s']}s\n\n"
        f"{emoji} *{avaliacao}*{warning_linha}\n\n"
    )

    # Detalha detecções por classe
    classes_count = {}
    for det in result["detections"]:
        cls = det["class"]
        classes_count[cls] = classes_count.get(cls, 0) + 1

    if classes_count:
        texto += "📋 *Detalhamento:*\n"
        class_emojis = {
            "Germinacao": "🌱",
            "Folha": "🍃",
        }
        class_display = {"Germinacao": "Germinação", "Folha": "Folha"}
        for cls, count in sorted(classes_count.items(), key=lambda x: -x[1]):
            emoji_cls = class_emojis.get(cls, "•")
            label = class_display.get(cls, cls)
            texto += f"  {emoji_cls} {label}: {count}\n"

    plantas = [d for d in result["detections"] if d.get("plant_id")]
    if plantas:
        texto += "\n🌱 *Por planta:*\n"
        for p in sorted(plantas, key=lambda x: x["plant_id"]):
            folhas = p["leaf_count"]
            texto += f"  #{p['plant_id']}: {folhas} folha{'s' if folhas != 1 else ''} ({p['confidence']*100:.0f}%)\n"

    # Envia a imagem resultado com legenda
    result_image_path = Path(current_app.root_path).parent / result["result_image"].lstrip("/")
    if result_image_path.exists():
        try:
            img_bytes = result_image_path.read_bytes()
            # Evolution API espera base64 puro, sem prefixo data:image/...;base64,
            img_b64 = base64.b64encode(img_bytes).decode()
            client.send_image_base64(sender, img_b64, filename="resultado.jpg", caption=texto)
        except Exception:
            traceback.print_exc()
            client.send_text(sender, texto)
    else:
        client.send_text(sender, texto)


def _handle_status_command(client, sender: str):
    """Retorna estatísticas gerais."""
    from flask import current_app
    import sqlite3

    db = current_app.config["DB_PATH"]
    try:
        with sqlite3.connect(db) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("""
                SELECT
                    COUNT(*) AS total,
                    ROUND(AVG(germination_rate), 1) AS avg_rate,
                    ROUND(MAX(germination_rate), 1) AS best_rate,
                    ROUND(MIN(germination_rate), 1) AS worst_rate,
                    ROUND(AVG(leaf_avg), 1) AS avg_leaves
                FROM analyses
            """).fetchone()

        if row and row["total"] > 0:
            texto = (
                f"📊 *Estatísticas — GerminaVision*\n\n"
                f"• Total de análises: {row['total']}\n"
                f"• Taxa média de germinação: {row['avg_rate']}%\n"
                f"• Melhor taxa: {row['best_rate']}%\n"
                f"• Pior taxa: {row['worst_rate']}%\n"
                f"• Média de folhas: {row['avg_leaves']}\n\n"
                f"📷 _Envie uma foto para nova análise!_"
            )
        else:
            texto = "📊 *Nenhuma análise registrada ainda.*\n\n📷 Envie uma foto de bandeja para começar!"

    except Exception as e:
        texto = f"❌ Erro ao consultar estatísticas: {e}"

    client.send_text(sender, texto)


def _handle_tip_command(client, sender: str):
    """Envia dica aleatória."""
    import random
    from app.chatbot import DICAS_GERMINACAO

    dica = random.choice(DICAS_GERMINACAO)
    client.send_text(sender, f"💡 *Dica do GerminaVision:*\n\n{dica}")


def _handle_history_command(client, sender: str):
    """Retorna últimas 5 análises."""
    from flask import current_app
    from app.database import get_history

    records = get_history(current_app.config["DB_PATH"], limit=5)

    if not records:
        client.send_text(sender, "📋 *Nenhuma análise no histórico.*\n\n📷 Envie uma foto para começar!")
        return

    texto = "📋 *Últimas análises — GerminaVision*\n\n"
    for i, r in enumerate(records, 1):
        texto += (
            f"*{i}.* {r['timestamp']}\n"
            f"   Germinação: {r['germination_rate']}% | "
            f"Mudas: {r['germinated']}/{r['total_detected']} | "
            f"Folhas: {r['leaf_avg']}\n\n"
        )

    client.send_text(sender, texto)


def _handle_help_command(client, sender: str):
    """Envia menu de ajuda."""
    texto = (
        "🌱 *GerminaVision — Menu de Ajuda*\n\n"
        "📷 *Envie uma foto* — Análise automática da bandeja\n"
        "📊 *status* — Estatísticas gerais\n"
        "📋 *histórico* — Últimas 5 análises\n"
        "💡 *dica* — Dica de germinação\n"
        "❓ *ajuda* — Este menu\n\n"
        "💬 Ou simplesmente *digite qualquer pergunta* sobre germinação "
        "e nosso GerminaBot com IA responderá!\n\n"
        "_Desenvolvido com visão computacional YOLO11_ 🤖"
    )
    client.send_text(sender, texto)


def _handle_chat_message(client, sender: str, text: str, quoted: str | None = None):
    """Passa mensagem para o GerminaBot (IA) com indicador 'digitando...' e memória de conversa."""
    from flask import current_app
    from app.chatbot import gerar_resposta

    # Dispara "digitando..." imediatamente. O delay 5000ms cobre a chamada LLM (~3-8s).
    client.send_presence(sender, "composing", delay_ms=5000)

    if quoted:
        quoted_short = quoted[:300] + ("..." if len(quoted) > 300 else "")
        composed = f'[Respondendo a sua mensagem anterior: "{quoted_short}"]\n{text}'
    else:
        composed = text

    resposta = gerar_resposta(composed, current_app.config["DB_PATH"], sender=sender)

    # Encerra o indicador antes de mandar a resposta
    client.send_presence(sender, "paused", delay_ms=0)
    client.send_text(sender, resposta)
