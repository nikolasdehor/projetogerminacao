"""Rotas do WhatsApp — webhook + painel de configuração."""
from __future__ import annotations

import base64
import json
import os
import time as _time
import traceback
from pathlib import Path

from flask import (
    Blueprint, current_app, jsonify, render_template, request
)

from app.whatsapp import get_client

wp = Blueprint("whatsapp", __name__)


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
        try:
            _handle_message(payload)
        except Exception as e:
            print(f"❌ Erro ao processar mensagem WhatsApp: {e}")
            traceback.print_exc()

    elif event in ("CONNECTION.UPDATE", "CONNECTION_UPDATE"):
        state = payload.get("data", {}).get("state", "unknown")
        print(f"📱 WhatsApp conexão: {state}")

    # Sempre retorna 200 para a Evolution API não reenviar
    return jsonify({"received": True}), 200


# ── Handler de mensagens ──────────────────────────────────────────────────────

def _handle_message(payload: dict):
    """Processa uma mensagem recebida pelo WhatsApp."""
    from flask import current_app

    data = payload.get("data", {})
    key = data.get("key", {})

    # Ignora mensagens enviadas por nós mesmos
    if key.get("fromMe", False):
        return

    remote_jid = key.get("remoteJid", "")
    # Extrai número limpo (5511999999999)
    sender = remote_jid.split("@")[0] if "@" in remote_jid else remote_jid

    if not sender:
        return

    msg = data.get("message", {})
    client = get_client()

    # ── Imagem recebida → roda inferência ──────────────────────────────────
    if "imageMessage" in msg or "documentMessage" in msg:
        _handle_image_message(client, sender, payload)
        return

    # ── Texto recebido → chatbot ou comandos ───────────────────────────────
    text = (
        msg.get("conversation")
        or msg.get("extendedTextMessage", {}).get("text")
        or ""
    ).strip().lower()

    if not text:
        return

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
        # Passa para o GerminaBot (IA)
        _handle_chat_message(client, sender, text)


def _handle_image_message(client, sender: str, payload: dict):
    """Processa imagem: baixa, roda YOLO, envia resultado."""
    from flask import current_app
    from app.inference import run_inference
    from app.database import insert_analysis

    upload_dir = current_app.config["UPLOAD_FOLDER"]
    result_dir = current_app.config["RESULT_FOLDER"]

    # Notifica que estamos processando
    client.send_text(sender, "🔬 *Analisando sua imagem...*\nAguarde um momento!")

    # Baixa a imagem
    image_path = client.download_media(payload, upload_dir)
    if not image_path:
        client.send_text(sender, "⚠️ Não consegui baixar a imagem. Tente enviar novamente.")
        return

    # Roda inferência
    try:
        result = run_inference(
            image_path=image_path,
            model=current_app.config["MODEL"],
            result_folder=result_dir,
            conf_threshold=0.25,
        )
    except Exception as e:
        client.send_text(sender, f"❌ Erro na análise: {e}")
        return

    # Salva no banco
    try:
        insert_analysis(
            db_path=current_app.config["DB_PATH"],
            filename=f"whatsapp_{sender}",
            total_detected=result["total_detected"],
            germinated=result["germinated"],
            germination_rate=result["germination_rate"],
            leaf_avg=result["leaf_avg"],
            result_image=result["result_image"],
            day_label=None,
        )
    except Exception:
        pass  # Não impede a resposta

    # Monta resposta
    rate = result["germination_rate"]
    if rate >= 80:
        emoji = "🟢"
        avaliacao = "Excelente! Retorno comercial garantido."
    elif rate >= 60:
        emoji = "🟡"
        avaliacao = "Moderado. Viável, mas com perda de eficiência."
    else:
        emoji = "🔴"
        avaliacao = "Atenção! Risco de prejuízo. Avalie as sementes e substrato."

    texto = (
        f"🌱 *Análise da Bandeja — GerminaVision*\n\n"
        f"📊 *Resultados:*\n"
        f"• Mudas detectadas: {result['total_detected']}\n"
        f"• Germinadas: {result['germinated']} ({rate}%)\n"
        f"• Folhas por muda (média): {result['leaf_avg']}\n"
        f"• Tempo de análise: {result['inference_time_s']}s\n\n"
        f"{emoji} *{avaliacao}*\n\n"
    )

    # Detalha detecções por classe
    classes_count = {}
    for det in result["detections"]:
        cls = det["class"]
        classes_count[cls] = classes_count.get(cls, 0) + 1

    if classes_count:
        texto += "📋 *Detalhamento:*\n"
        class_emojis = {
            "seedling": "🌿", "twoseedling": "🌿🌿", "weak": "😟",
            "noseedling": "❌", "processed": "✅", "askew": "↗️",
        }
        for cls, count in sorted(classes_count.items(), key=lambda x: -x[1]):
            emoji_cls = class_emojis.get(cls, "•")
            texto += f"  {emoji_cls} {cls}: {count}\n"

    # Envia a imagem resultado com legenda
    result_image_path = Path(current_app.root_path).parent / result["result_image"].lstrip("/")
    if result_image_path.exists():
        try:
            img_bytes = result_image_path.read_bytes()
            img_b64 = "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode()
            client.send_image_base64(sender, img_b64, caption=texto)
        except Exception:
            # Fallback: envia só texto
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


def _handle_chat_message(client, sender: str, text: str):
    """Passa mensagem para o GerminaBot (IA)."""
    from flask import current_app
    from app.chatbot import gerar_resposta

    resposta = gerar_resposta(text, current_app.config["DB_PATH"])
    client.send_text(sender, resposta)
