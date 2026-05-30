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

from app.rate_limit import get_rate_limit_store

# Dedup de message_id para descartar reentregas da Evolution API
_seen_msg_ids: deque[str] = deque(maxlen=500)
_seen_ids_lock = Lock()

GROUP_RESPONSE_MODE = os.getenv("GROUP_RESPONSE_MODE", "image_always_text_mention").lower()
ALLOWED_GROUPS = [
    g.strip() for g in os.getenv("ALLOWED_GROUPS", "").split(",") if g.strip()
]
_VALID_GROUP_RESPONSE_MODES = {"mention_only", "image_always_text_mention", "all", "off"}

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
            process_webhook_message(payload)
    except Exception as exc:
        app.logger.exception(f"[worker] erro processando mensagem: {exc}")
    finally:
        with _queue_lock:
            _queue_pending["count"] = max(0, _queue_pending["count"] - 1)

from app.whatsapp import get_client
from app.processed_messages import get_store
from app.inference import parse_caption as _parse_caption
from app.guards import should_process_image, passes_post_inference_guard, _jid_truncated

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


def _group_response_mode() -> str:
    mode = os.getenv("GROUP_RESPONSE_MODE", GROUP_RESPONSE_MODE).strip().lower()
    if mode not in _VALID_GROUP_RESPONSE_MODES:
        return "image_always_text_mention"
    return mode


def _allowed_groups() -> list[str]:
    raw = os.getenv("ALLOWED_GROUPS")
    if raw is None:
        return ALLOWED_GROUPS
    return [g.strip() for g in raw.split(",") if g.strip()]


def _extract_message_text(msg: dict) -> str:
    if not isinstance(msg, dict):
        return ""
    return (
        msg.get("conversation")
        or msg.get("extendedTextMessage", {}).get("text")
        or msg.get("imageMessage", {}).get("caption")
        or msg.get("documentMessage", {}).get("caption")
        or ""
    ).strip()


def _message_contexts(msg: dict) -> list[dict]:
    if not isinstance(msg, dict):
        return []
    contexts = []
    for msg_type in ("extendedTextMessage", "imageMessage", "documentMessage"):
        ctx = msg.get(msg_type, {}).get("contextInfo", {})
        if isinstance(ctx, dict) and ctx:
            contexts.append(ctx)
    return contexts


def _has_bot_mention(msg: dict, text: str, bot_phone: str) -> bool:
    if not bot_phone:
        return False

    bot_phone = bot_phone.strip().lower()
    if f"@{bot_phone}" in (text or "").lower():
        return True

    for ctx in _message_contexts(msg):
        mentioned_jids = ctx.get("mentionedJid", [])
        if isinstance(mentioned_jids, str):
            mentioned_jids = [mentioned_jids]
        if any(bot_phone in str(mentioned_jid).lower() for mentioned_jid in mentioned_jids):
            return True

        quoted_from = ctx.get("participant", "")
        if bot_phone in str(quoted_from).lower():
            return True

    return False


def _strip_bot_mention(text: str, bot_phone: str) -> str:
    if not text or not bot_phone:
        return text
    return (
        text.replace(f"@{bot_phone}", "")
        .replace(f"@+{bot_phone}", "")
        .strip()
    )


def _sender_short_label(sender: str) -> str:
    sender = sender.strip()
    if not sender:
        return "usuario"
    if len(sender) <= 8:
        return f"+{sender}"
    return f"+{sender[:4]}...{sender[-4:]}"


# ── Página de configuração ────────────────────────────────────────────────────

@wp.route("/whatsapp")
def whatsapp_config():
    """Página de configuração do WhatsApp."""
    return render_template("whatsapp.html", v=int(_time.time()))


# ── API: Status da conexão ────────────────────────────────────────────────────

@wp.route("/api/whatsapp/status")
def whatsapp_status():
    """Retorna status da conexão WhatsApp."""
    instance_name = os.getenv("EVOLUTION_INSTANCE_NAME", "")
    client = get_client()

    # Verifica se a PUBLIC_BASE_URL é realmente pública (não localhost/127.x)
    public_base_url = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    _local_patterns = ("localhost", "127.0.0.1", "0.0.0.0", "::1")
    public_url_ok = bool(public_base_url) and not any(p in public_base_url for p in _local_patterns)

    if not client.is_configured():
        return jsonify({
            "configured": False,
            "state": "unconfigured",
            "instance": instance_name,
            "public_url_ok": public_url_ok,
            "message": "Evolution API não configurada. Preencha as variáveis no .env",
        })

    try:
        status = client.get_instance_status()
        state = status.get("instance", {}).get("state", status.get("state", "close"))
        return jsonify({
            "configured": True,
            "state": state,
            "instance": instance_name,
            "public_url_ok": public_url_ok,
            "message": _state_message(state),
        })
    except RuntimeError as e:
        return jsonify({
            "configured": True,
            "state": "error",
            "instance": instance_name,
            "public_url_ok": public_url_ok,
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
    """Cria instância e retorna QR Code. Toda config vem do .env."""
    client = get_client()
    if not client.is_configured():
        return jsonify({"error": "Evolution API não configurada"}), 400

    public_url = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    if not public_url:
        public_url = f"https://{request.host}" if request.host else ""
    webhook_url = f"{public_url}/api/whatsapp/webhook" if public_url else None

    try:
        result = client.create_instance(webhook_url=webhook_url)
        qrcode_data = result.get("qrcode", {})

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

@wp.route("/api/whatsapp/qr")
def whatsapp_qr():
    """Retorna QR Code atualizado para conexão (formato Evolution API)."""
    client = get_client()
    try:
        result = client.get_qrcode()
        state = "close"
        try:
            status = client.get_instance_status()
            state = status.get("instance", {}).get("state", status.get("state", "close"))
        except RuntimeError:
            pass
        return jsonify({
            "base64": result.get("base64", ""),
            "pairingCode": result.get("pairingCode", ""),
            "state": state,
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
        message_id = _message_id(payload)
        store = get_store()
        if message_id and store.is_processed(message_id):
            return jsonify({"ok": True, "skipped": "already_processed"}), 200

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

def _message_id(message_data: dict) -> str:
    return (
        message_data.get("data", {}).get("key", {}).get("id")
        or message_data.get("data", {}).get("messageId")
        or ""
    )


def _message_timestamp(message_data: dict) -> int | None:
    msg_ts = message_data.get("data", {}).get("messageTimestamp")
    try:
        return int(msg_ts) if msg_ts is not None else None
    except (TypeError, ValueError):
        return None


def process_webhook_message(message_data: dict) -> None:
    """Processa uma message_data e marca como concluida no dedupe persistente."""
    message_id = _message_id(message_data)
    store = get_store()
    if message_id and store.is_processed(message_id):
        current_app.logger.info(f"[dedup] message_id ja processado: {message_id}")
        return

    handled = _handle_message(message_data)
    if handled and message_id:
        store.mark_processed(message_id, _message_timestamp(message_data))


def _handle_message(payload: dict) -> bool:
    """Processa uma mensagem recebida pelo WhatsApp."""
    from flask import current_app

    data = payload.get("data", {})
    key = data.get("key", {})

    # Ignora mensagens enviadas por nós mesmos
    if key.get("fromMe", False):
        return False

    # Dedup thread-safe: descarta reentregas do mesmo webhook pela Evolution API
    msg_id = key.get("id", "")
    if msg_id:
        with _seen_ids_lock:
            if msg_id in _seen_msg_ids:
                current_app.logger.info(f"[dedup] message_id duplicado ignorado: {msg_id}")
                return False
            _seen_msg_ids.append(msg_id)

    remote_jid = key.get("remoteJid", "")
    participant_jid = key.get("participant", "")

    if remote_jid.endswith("@g.us"):
        chat_type = "group"
        group_jid = remote_jid
        # Em grupo, o historico precisa ficar no usuario real, nao no JID do grupo.
        sender_jid = participant_jid or remote_jid
        sender = sender_jid.split("@")[0] if "@" in sender_jid else sender_jid
        reply_target = group_jid
    elif remote_jid.endswith("@broadcast"):
        return False
    else:
        chat_type = "dm"
        group_jid = None
        # Em DM, a Evolution espera o numero limpo como destino.
        sender = remote_jid.split("@")[0] if "@" in remote_jid else remote_jid
        reply_target = sender

    if not sender:
        return False

    msg = data.get("message", {})
    if not isinstance(msg, dict):
        return False

    raw_text = _extract_message_text(msg)
    is_image = bool(msg.get("imageMessage") or msg.get("documentMessage"))

    if chat_type == "group":
        allowed_groups = _allowed_groups()
        if allowed_groups and group_jid not in allowed_groups:
            return False

        mode = _group_response_mode()
        if mode == "off":
            return False

        bot_phone = os.getenv("EVOLUTION_INSTANCE_PHONE", "").strip()
        has_mention = _has_bot_mention(msg, raw_text, bot_phone)

        if mode == "mention_only" and not has_mention:
            return False
        if mode == "image_always_text_mention" and not is_image and not has_mention:
            return False

    # Filtra tipos de mensagem que não devem ser processados (reactions, protocol, etc.)
    _ALLOWED_MSG_TYPES = {"imageMessage", "conversation", "extendedTextMessage", "documentMessage"}
    if msg and not any(k in msg for k in _ALLOWED_MSG_TYPES):
        current_app.logger.info(f"[skip] tipo de mensagem ignorado: {list(msg.keys())}")
        return True

    if chat_type == "group":
        rate_store = get_rate_limit_store()
        if is_image:
            if rate_store.count_in_window(sender, "image", window_seconds=3600) >= 5:
                sender_short_label = _sender_short_label(sender)
                client = get_client()
                client.send_text(
                    reply_target,
                    f"{sender_short_label}: limite de 5 fotos/hora atingido. "
                    "Tente de novo daqui a pouco.",
                )
                return False
            rate_store.record(sender, "image")
        else:
            if rate_store.count_in_window(sender, "text", window_seconds=3600) >= 15:
                return False
            rate_store.record(sender, "text")

    client = get_client()

    # ── Imagem recebida → roda inferência ──────────────────────────────────
    if "imageMessage" in msg or "documentMessage" in msg:
        raw_caption = (
            msg.get("imageMessage", {}).get("caption")
            or msg.get("documentMessage", {}).get("caption")
            or ""
        ).strip() or None

        guard_msg = {"remoteJid": remote_jid, "caption": raw_caption or ""}
        process, guard_reason = should_process_image(guard_msg)
        if not process:
            jid_short = _jid_truncated(remote_jid)
            current_app.logger.info(
                "guard_skip group=%s reason=%s caption_snippet=%.40s",
                jid_short,
                guard_reason,
                raw_caption or "",
            )
            return True

        _handle_image_message(client, sender, reply_target, payload, raw_caption=raw_caption)
        return True

    # ── Texto recebido → chatbot ou comandos ───────────────────────────────
    if chat_type == "group":
        raw_text = _strip_bot_mention(raw_text, os.getenv("EVOLUTION_INSTANCE_PHONE", "").strip())
    text = raw_text.strip().lower()

    if not text:
        return True

    quoted = _extract_quoted_text(msg)

    # Comandos especiais
    if text in ("status", "estatísticas", "estatisticas", "stats"):
        _handle_status_command(client, reply_target)
    elif text in ("dica", "dicas", "tip"):
        _handle_tip_command(client, reply_target)
    elif text in ("histórico", "historico", "history"):
        _handle_history_command(client, reply_target)
    elif text in ("ajuda", "help", "menu", "?"):
        _handle_help_command(client, reply_target)
    else:
        # Passa para o GerminaBot (IA) com contexto de quote se houver
        _handle_chat_message(client, sender, reply_target, text, quoted=quoted)

    return True


def _handle_image_message(
    client,
    sender: str,
    reply_target: str,
    payload: dict,
    raw_caption: str | None = None,
) -> None:
    """Processa imagem: baixa, roda YOLO, envia resultado."""
    from flask import current_app
    from app.inference import run_inference
    from app.database import insert_analysis

    upload_dir = current_app.config["UPLOAD_FOLDER"]
    result_dir = current_app.config["RESULT_FOLDER"]

    caption, tray_capacity_override = _parse_caption(raw_caption)

    # Notifica que estamos processando (com indicador "digitando..." para feedback visual)
    client.send_presence(reply_target, "composing", delay_ms=3000)
    client.send_text(reply_target, "🌱 *Analisando a imagem...*\nEstou procurando plantas e folhas.")

    # Baixa a imagem com timeout explicito via executor isolado.
    # O Fluxo 3 do download_media pode travar em CLOSE_WAIT (CDN Meta fecha
    # TCP sem TLS close_notify). socket.setdefaulttimeout esta configurado em
    # download_media, mas usamos future.result(timeout=40) como segunda barreira.
    import concurrent.futures as _cf
    _dl_ex = _cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="wa-dl")
    try:
        _dl_fut = _dl_ex.submit(client.download_media, payload, upload_dir)
        try:
            image_path = _dl_fut.result(timeout=40)
        except _cf.TimeoutError:
            _dl_fut.cancel()
            client.send_text(reply_target, "⚠️ Tempo limite ao baixar a imagem (CDN lento). Tente novamente.")
            return
    finally:
        _dl_ex.shutdown(wait=False, cancel_futures=True)

    if not image_path:
        client.send_text(reply_target, "⚠️ Não consegui baixar a imagem. Tente enviar novamente.")
        return

    # Roda inferência com timeout de 120s para evitar worker preso indefinidamente.
    # SAHI em CPU pode levar ate 60s em fotos grandes; 120s da margem confortavel.
    _inf_ex = _cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="wa-inf")
    try:
        _inf_fut = _inf_ex.submit(
            run_inference,
            image_path=image_path,
            model=current_app.config["MODEL"],
            result_folder=result_dir,
            conf_threshold=0.5,
            tray_capacity_override=tray_capacity_override,
        )
        try:
            result = _inf_fut.result(timeout=120)
        except _cf.TimeoutError:
            _inf_fut.cancel()
            client.send_text(
                reply_target,
                "⏱️ A análise demorou mais que o esperado e foi cancelada.\n"
                "Tente com uma imagem menor ou com melhor iluminação.",
            )
            return
        except Exception as e:
            client.send_text(reply_target, f"❌ Erro na análise: {e}")
            return
    finally:
        _inf_ex.shutdown(wait=False, cancel_futures=True)

    detections = result.get("detections", [])
    mean_conf = (
        sum(d["confidence"] for d in detections) / len(detections) if detections else 0.0
    )
    passes, post_reason = passes_post_inference_guard(detections, mean_conf)
    if not passes:
        jid_short = _jid_truncated(reply_target)
        current_app.logger.info(
            "guard_skip group=%s reason=%s caption_snippet=%.40s",
            jid_short,
            post_reason,
            caption or "",
        )
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
            cells_count=result.get("cells_detected"),
            cells_origin=result.get("cells_origin"),
            rate_reliable=bool(result.get("rate_reliable", True)),
            rate_scope=result.get("rate_scope"),
        )
    except Exception:
        pass  # Não impede a resposta
    label_linha = f"🏷️ *Tratamento:* {caption}\n" if caption else ""
    cells_origin = result.get("cells_origin", "detected")
    rate_reliable = bool(result.get("rate_reliable", cells_origin != "fallback_default"))
    origem_label = {
        "caption": "total informado",
        "detected": "células detectadas",
        "detected_visible": "células visíveis",
        "fallback_default": "base padrão não confirmada",
    }.get(cells_origin, "células detectadas")
    cells_warning = result.get("cells_warning")
    quality_level = result.get("quality_level")
    quality_warning = result.get("quality_warning")

    if not rate_reliable:
        emoji = "⚪"
        if quality_level == "low":
            avaliacao = (
                "Leitura parcial: a iluminação da foto está crítica, então localizei o que foi possível, "
                "mas não vou classificar a taxa como confiável."
            )
        else:
            avaliacao = (
                "Leitura parcial: localizei as plantas visíveis, mas o recorte não dá uma base segura "
                "de células para calcular taxa."
            )
    elif cells_origin in ("detected", "detected_visible"):
        if rate >= 75:
            emoji = "🟢"
            avaliacao = "Boa germinação no recorte visível. Use como leitura da área fotografada."
        elif rate >= 55:
            emoji = "🟡"
            avaliacao = "Germinação moderada no recorte visível. Confira a imagem anotada e, se possível, envie a bandeja inteira."
        else:
            emoji = "🟡"
            avaliacao = (
                "Leitura do recorte visível: a taxa indica quantas células apresentaram muda "
                "na área fotografada, não na bandeja inteira."
            )
    elif rate >= 75:
        emoji = "🟢"
        avaliacao = "Boa germinação na base informada. Continue acompanhando uniformidade e crescimento."
    elif rate >= 55:
        emoji = "🟡"
        avaliacao = "Germinação moderada na base informada. Vale observar espaços vazios e repetir a foto depois."
    else:
        emoji = "🔴"
        avaliacao = (
            "Taxa baixa na base informada. Confira sementes, substrato, umidade e temperatura, "
            "mas confirme se a foto cobre a bandeja inteira."
        )

    if not rate_reliable:
        plants_line = (
            f"• Plantas localizadas: {germinated}\n"
            f"• Taxa da bandeja: não confirmada (leitura parcial)\n"
        )
    elif cells_origin == "caption":
        plants_line = f"• Plantas germinadas: {germinated} de {capacity} células ({rate}%) [{origem_label}]\n"
    elif cells_origin in ("detected", "detected_visible"):
        plants_line = f"• No recorte: {germinated} plantas em {capacity} células visíveis ({rate}%)\n"
    else:
        plants_line = (
            f"• Plantas germinadas detectadas: {germinated}\n"
            f"• Taxa da bandeja: não confirmada (células visíveis não contadas com segurança)\n"
        )

    warnings = []
    if quality_warning:
        warnings.append(f"_⚠️ {quality_warning}_")
    if cells_warning:
        warnings.append(cells_warning)
    elif cells_origin in ("detected", "detected_visible") and rate_reliable:
        warnings.append("_ℹ️ Taxa calculada só sobre o recorte visível, não sobre a bandeja inteira._")
    warning_linha = "\n" + "\n".join(warnings) if warnings else ""

    # Texto adaptativo: se detectou bandeja confiavelmente, fala "bandeja";
    # senao, fala "imagem". Mantem linguagem natural ao inves de hardcoded.
    tem_bandeja = rate_reliable and cells_origin in ("caption", "detected", "detected_visible")
    titulo_objeto = "bandeja" if tem_bandeja else "imagem"

    texto = (
        f"🌱 *GerminaVision — Análise da {titulo_objeto}*\n"
        f"{label_linha}\n"
        f"📊 *Resultados:*\n"
        f"{plants_line}"
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
        texto += "🔎 *Leitura técnica:*\n"
        class_emojis = {
            "Germinacao": "🌱",
            "Folha": "🍃",
        }
        class_display = {
            "Germinacao": "Plantas localizadas",
            "Folha": "Sinais de folha detectados",
        }
        class_order = {"Germinacao": 0, "Folha": 1}
        for cls, count in sorted(classes_count.items(), key=lambda x: (class_order.get(x[0], 99), x[0])):
            emoji_cls = class_emojis.get(cls, "•")
            label = class_display.get(cls, cls)
            texto += f"  {emoji_cls} {label}: {count}\n"

    plantas = [d for d in result["detections"] if d.get("plant_id")]
    if plantas:
        texto += "\n🌱 *Por planta:*\n"
        for p in sorted(plantas, key=lambda x: x["plant_id"]):
            folhas = p["leaf_count"]
            folhas_txt = (
                "folhas não estimadas"
                if folhas <= 0
                else f"{folhas} folha{'s' if folhas != 1 else ''}"
            )
            texto += f"  #{p['plant_id']}: {folhas_txt} ({p['confidence']*100:.0f}%)\n"

    # Envia a imagem resultado com legenda
    result_image_path = Path(current_app.root_path).parent / result["result_image"].lstrip("/")
    if result_image_path.exists():
        try:
            img_bytes = result_image_path.read_bytes()
            # Evolution API espera base64 puro, sem prefixo data:image/...;base64,
            img_b64 = base64.b64encode(img_bytes).decode()
            client.send_image_base64(reply_target, img_b64, filename="resultado.jpg", caption=texto)
        except Exception:
            traceback.print_exc()
            client.send_text(reply_target, texto)
    else:
        client.send_text(reply_target, texto)


def _handle_status_command(client, reply_target: str) -> None:
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
                    SUM(CASE WHEN COALESCE(rate_reliable, 1) = 1 THEN 1 ELSE 0 END) AS reliable_total,
                    ROUND(AVG(CASE WHEN COALESCE(rate_reliable, 1) = 1 THEN germination_rate END), 1) AS avg_rate,
                    ROUND(MAX(CASE WHEN COALESCE(rate_reliable, 1) = 1 THEN germination_rate END), 1) AS best_rate,
                    ROUND(MIN(CASE WHEN COALESCE(rate_reliable, 1) = 1 THEN germination_rate END), 1) AS worst_rate,
                    ROUND(AVG(leaf_avg), 1) AS avg_leaves
                FROM analyses
            """).fetchone()

        if row and row["total"] > 0:
            avg_rate = f"{row['avg_rate']}%" if row["avg_rate"] is not None else "—"
            best_rate = f"{row['best_rate']}%" if row["best_rate"] is not None else "—"
            worst_rate = f"{row['worst_rate']}%" if row["worst_rate"] is not None else "—"
            texto = (
                f"📊 *Estatísticas — GerminaVision*\n\n"
                f"• Total de análises: {row['total']}\n"
                f"• Análises com taxa confiável: {row['reliable_total']}\n"
                f"• Taxa média de germinação: {avg_rate}\n"
                f"• Melhor taxa: {best_rate}\n"
                f"• Pior taxa: {worst_rate}\n"
                f"• Média de folhas: {row['avg_leaves']}\n\n"
                f"📷 _Envie uma foto para nova análise!_"
            )
        else:
            texto = "📊 *Nenhuma análise registrada ainda.*\n\n📷 Envie uma foto para começar!"

    except Exception as e:
        texto = f"❌ Erro ao consultar estatísticas: {e}"

    client.send_text(reply_target, texto)


def _handle_tip_command(client, reply_target: str) -> None:
    """Envia dica aleatória."""
    import random
    from app.chatbot import DICAS_GERMINACAO

    dica = random.choice(DICAS_GERMINACAO)
    client.send_text(reply_target, f"💡 *Dica do GerminaVision:*\n\n{dica}")


def _handle_history_command(client, reply_target: str) -> None:
    """Retorna últimas 5 análises."""
    from flask import current_app
    from app.database import get_history

    records = get_history(current_app.config["DB_PATH"], limit=5)

    if not records:
        client.send_text(reply_target, "📋 *Nenhuma análise no histórico.*\n\n📷 Envie uma foto para começar!")
        return

    texto = "📋 *Últimas análises — GerminaVision*\n\n"
    for i, r in enumerate(records, 1):
        texto += (
            f"*{i}.* {r['timestamp']}\n"
            f"   Germinação: {r['germination_rate']}% | "
            f"Mudas: {r['germinated']}/{r['total_detected']} | "
            f"Folhas: {r['leaf_avg']}\n\n"
        )

    client.send_text(reply_target, texto)


def _handle_help_command(client, reply_target: str) -> None:
    """Envia menu de ajuda."""
    texto = (
        "🌱 *GerminaVision — Menu de Ajuda*\n\n"
        "📷 *Envie uma foto* — Análise automática de plantas e folhas\n"
        "📊 *status* — Estatísticas gerais\n"
        "📋 *histórico* — Últimas 5 análises\n"
        "💡 *dica* — Dica de germinação\n"
        "❓ *ajuda* — Este menu\n\n"
        "💬 Ou simplesmente *digite qualquer pergunta* sobre germinação "
        "e nosso GerminaBot com IA responderá!\n\n"
        "_Desenvolvido com visão computacional YOLO11_ 🤖"
    )
    client.send_text(reply_target, texto)


def _handle_chat_message(
    client,
    sender: str,
    reply_target: str,
    text: str,
    quoted: str | None = None,
) -> None:
    """Passa mensagem para o GerminaBot (IA) com indicador 'digitando...' e memória de conversa."""
    from flask import current_app
    from app.chatbot import gerar_resposta

    # Dispara "digitando..." imediatamente. O delay 5000ms cobre a chamada LLM (~3-8s).
    client.send_presence(reply_target, "composing", delay_ms=5000)

    if quoted:
        quoted_short = quoted[:300] + ("..." if len(quoted) > 300 else "")
        composed = f'[Respondendo a sua mensagem anterior: "{quoted_short}"]\n{text}'
    else:
        composed = text

    resposta = gerar_resposta(composed, current_app.config["DB_PATH"], sender=sender)

    # Encerra o indicador antes de mandar a resposta
    client.send_presence(reply_target, "paused", delay_ms=0)
    client.send_text(reply_target, resposta)
