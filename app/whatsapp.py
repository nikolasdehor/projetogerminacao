"""Cliente da Evolution API — ponte entre GerminaVision e WhatsApp."""
from __future__ import annotations

import io
import json
import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional


class EvolutionClient:
    """Wrapper minimalista para a Evolution API v2."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        instance_name: Optional[str] = None,
    ):
        self.base_url = (base_url or os.getenv("EVOLUTION_API_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("EVOLUTION_API_KEY", "")
        self.instance_name = instance_name or os.getenv("EVOLUTION_INSTANCE_NAME", "germinavision")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "apikey": self.api_key,
        }

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        """Faz request para a Evolution API e retorna JSON."""
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body else None
        req = urllib.request.Request(url, data=data, headers=self._headers(), method=method)

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Evolution API {e.code}: {error_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Falha de conexão com Evolution API: {e}") from e

    def is_configured(self) -> bool:
        """Retorna True se as variáveis da Evolution API estão configuradas."""
        return bool(self.base_url and self.api_key)

    # ── Instância ──────────────────────────────────────────────────────────────

    def create_instance(self, webhook_url: str) -> dict:
        """Cria ou recria a instância do WhatsApp com webhook configurado."""
        return self._request("POST", "/instance/create", {
            "instanceName": self.instance_name,
            "integration": "WHATSAPP-BAILEYS",
            "qrcode": True,
            "webhook": {
                "url": webhook_url,
                "byEvents": False,
                "base64": True,
                "events": [
                    "MESSAGES_UPSERT",
                    "CONNECTION_UPDATE",
                ],
            },
        })

    def get_instance_status(self) -> dict:
        """Retorna status da instância (open, close, connecting)."""
        try:
            result = self._request("GET", f"/instance/connectionState/{self.instance_name}")
            return result
        except RuntimeError:
            return {"state": "close"}

    def get_qrcode(self) -> dict:
        """Retorna QR Code para conectar o WhatsApp."""
        return self._request("GET", f"/instance/connect/{self.instance_name}")

    def logout_instance(self) -> dict:
        """Desconecta o WhatsApp da instância (mantém a instância ativa)."""
        return self._request("DELETE", f"/instance/logout/{self.instance_name}")

    def delete_instance(self) -> dict:
        """Remove a instância por completo."""
        return self._request("DELETE", f"/instance/delete/{self.instance_name}")

    def restart_instance(self) -> dict:
        """Reinicia a instância."""
        return self._request("PUT", f"/instance/restart/{self.instance_name}")

    # ── Mensagens ──────────────────────────────────────────────────────────────

    def send_text(self, to: str, text: str) -> dict:
        """Envia mensagem de texto. `to` deve ser no formato 5511999999999."""
        return self._request("POST", f"/message/sendText/{self.instance_name}", {
            "number": to,
            "text": text,
        })

    def send_presence(self, to: str, presence: str = "composing", delay_ms: int = 1200) -> dict:
        """
        Envia indicador de presença ("digitando..." aparece no WhatsApp do destinatário).
        presence: 'composing' (digitando), 'recording' (gravando áudio), 'paused' (parou), 'available' (online).
        Falha silenciosamente para não interromper a resposta principal.
        """
        try:
            return self._request("POST", f"/chat/sendPresence/{self.instance_name}", {
                "number": to,
                "delay": delay_ms,
                "presence": presence,
            })
        except RuntimeError:
            return {}

    def send_image(self, to: str, image_url: str, caption: str = "") -> dict:
        """Envia imagem com legenda. `image_url` deve ser URL pública da imagem."""
        return self._request("POST", f"/message/sendMedia/{self.instance_name}", {
            "number": to,
            "mediatype": "image",
            "media": image_url,
            "caption": caption,
        })

    def send_image_base64(self, to: str, base64_data: str, filename: str = "resultado.jpg", caption: str = "") -> dict:
        """Envia imagem como base64."""
        return self._request("POST", f"/message/sendMedia/{self.instance_name}", {
            "number": to,
            "mediatype": "image",
            "media": base64_data,
            "fileName": filename,
            "caption": caption,
        })

    # ── Download de mídia ──────────────────────────────────────────────────────

    def download_media(self, message_data: dict, save_dir: str) -> Optional[str]:
        """
        Baixa mídia (imagem) de uma mensagem recebida pelo webhook.
        Tenta 3 fluxos em ordem: base64 no webhook → getBase64FromMediaMessage → URL direta.
        Retorna o caminho do arquivo salvo, ou None se falhar.
        """
        import base64 as b64
        import uuid

        msg = message_data.get("data", {}).get("message", {})
        image_msg = msg.get("imageMessage") or msg.get("documentMessage") or {}
        ext = image_msg.get("mimetype", "image/jpeg").split("/")[-1]
        if ext not in ("jpeg", "jpg", "png", "webp"):
            ext = "jpg"

        raw: Optional[bytes] = None

        # Fluxo 1: base64 enviado diretamente no webhook (requer base64:true na config)
        base64_data = msg.get("base64") or message_data.get("data", {}).get("message", {}).get("base64")
        if base64_data:
            # Remove prefixo data:image/...;base64, se presente
            if "," in base64_data:
                base64_data = base64_data.split(",", 1)[1]
            try:
                raw = b64.b64decode(base64_data)
            except Exception:
                raw = None

        # Fluxo 2: buscar base64 via endpoint da Evolution API
        if not raw:
            try:
                key = message_data.get("data", {}).get("key", {})
                resp = self._request(
                    "POST",
                    f"/chat/getBase64FromMediaMessage/{self.instance_name}",
                    {"message": {"key": key}},
                )
                b64_str = resp.get("base64", "")
                if b64_str:
                    if "," in b64_str:
                        b64_str = b64_str.split(",", 1)[1]
                    raw = b64.b64decode(b64_str)
            except Exception:
                raw = None

        # Fluxo 3: download direto via URL (com apikey no header)
        if not raw:
            media_url = image_msg.get("url") or image_msg.get("directPath")
            if media_url:
                try:
                    req = urllib.request.Request(media_url, headers={"apikey": self.api_key})
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        raw = resp.read()
                except Exception:
                    raw = None

        if not raw:
            return None

        # Valida magic bytes antes de salvar
        _VALID_MAGIC = (b"\xff\xd8", b"\x89PNG", b"GIF8", b"RIFF", b"WEBP")
        if not any(raw[:4].startswith(m) for m in _VALID_MAGIC):
            return None

        # Confirma com PIL
        try:
            from PIL import Image
            Image.open(io.BytesIO(raw)).verify()
        except Exception:
            return None

        filename = f"wa_{uuid.uuid4().hex[:12]}.{ext}"
        save_path = Path(save_dir) / filename
        save_path.write_bytes(raw)
        return str(save_path)


# ── Singleton global ───────────────────────────────────────────────────────────

_client: Optional[EvolutionClient] = None


def get_client() -> EvolutionClient:
    """Retorna instância singleton do cliente Evolution API."""
    global _client
    if _client is None:
        _client = EvolutionClient()
    return _client
