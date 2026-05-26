"""Testes de tratamento de mensagens em grupos vs DM."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from flask import Flask

from app import whatsapp_routes


class FakeRateLimitStore:
    def __init__(self) -> None:
        self.counts: dict[tuple[str, str], int] = {}
        self.recorded: list[tuple[str, str]] = []

    def count_in_window(self, sender: str, kind: str, window_seconds: int = 3600) -> int:
        return self.counts.get((sender, kind), 0)

    def record(self, sender: str, kind: str) -> None:
        self.recorded.append((sender, kind))
        self.counts[(sender, kind)] = self.counts.get((sender, kind), 0) + 1


class GroupChatRoutingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        base = Path(self.tmpdir.name)
        self.app = Flask(__name__)
        self.app.config.update(
            DB_PATH=str(base / "test.db"),
            UPLOAD_FOLDER=str(base / "uploads"),
            RESULT_FOLDER=str(base / "results"),
            MODEL=None,
        )
        Path(self.app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
        Path(self.app.config["RESULT_FOLDER"]).mkdir(parents=True, exist_ok=True)
        self.ctx = self.app.app_context()
        self.ctx.push()

        with whatsapp_routes._seen_ids_lock:
            whatsapp_routes._seen_msg_ids.clear()

        self.bot_phone = "5562998561249"
        self.client = MagicMock()
        self.rate_store = FakeRateLimitStore()
        self.env_patch = patch.dict(
            os.environ,
            {
                "EVOLUTION_INSTANCE_PHONE": self.bot_phone,
                "GROUP_RESPONSE_MODE": "image_always_text_mention",
                "ALLOWED_GROUPS": "",
            },
            clear=False,
        )
        self.get_client_patch = patch.object(whatsapp_routes, "get_client", return_value=self.client)
        self.rate_patch = patch.object(
            whatsapp_routes,
            "get_rate_limit_store",
            return_value=self.rate_store,
        )
        self.env_patch.start()
        self.get_client_patch.start()
        self.rate_patch.start()

    def tearDown(self) -> None:
        self.rate_patch.stop()
        self.get_client_patch.stop()
        self.env_patch.stop()
        with whatsapp_routes._seen_ids_lock:
            whatsapp_routes._seen_msg_ids.clear()
        self.ctx.pop()
        self.tmpdir.cleanup()

    def _payload_dm(self, msg_id: str = "ABC123", text: str = "oi") -> dict:
        return {
            "event": "MESSAGES.UPSERT",
            "data": {
                "key": {
                    "remoteJid": "5511999999999@s.whatsapp.net",
                    "fromMe": False,
                    "id": msg_id,
                },
                "messageTimestamp": 1700000000,
                "message": {"conversation": text},
            },
        }

    def _payload_group(
        self,
        msg_id: str = "GRP1",
        text: str = "oi",
        with_mention: bool = False,
        bot_phone: str | None = None,
    ) -> dict:
        bot_phone = bot_phone or self.bot_phone
        msg = {"conversation": text}
        if with_mention:
            msg = {
                "extendedTextMessage": {
                    "text": f"@{bot_phone} {text}",
                    "contextInfo": {"mentionedJid": [f"{bot_phone}@s.whatsapp.net"]},
                }
            }
        return {
            "event": "MESSAGES.UPSERT",
            "data": {
                "key": {
                    "remoteJid": "120363025481-12345@g.us",
                    "fromMe": False,
                    "id": msg_id,
                    "participant": "5511888888888@s.whatsapp.net",
                },
                "messageTimestamp": 1700000000,
                "message": msg,
            },
        }

    def _payload_group_image(self, msg_id: str = "IMG1") -> dict:
        payload = self._payload_group(msg_id=msg_id)
        payload["data"]["message"] = {"imageMessage": {"url": "https://example.test/image.jpg"}}
        return payload

    def test_dm_text_processes_normally(self) -> None:
        with patch("app.chatbot.gerar_resposta", return_value="resposta") as gerar_resposta:
            handled = whatsapp_routes._handle_message(self._payload_dm())

        self.assertTrue(handled)
        gerar_resposta.assert_called_once()
        self.assertEqual(gerar_resposta.call_args.kwargs["sender"], "5511999999999")
        self.client.send_text.assert_called_once_with("5511999999999", "resposta")

    def test_group_text_without_mention_ignored(self) -> None:
        with patch("app.chatbot.gerar_resposta", return_value="resposta") as gerar_resposta:
            handled = whatsapp_routes._handle_message(self._payload_group())

        self.assertFalse(handled)
        gerar_resposta.assert_not_called()
        self.client.send_text.assert_not_called()
        self.assertEqual(self.rate_store.recorded, [])

    def test_group_image_always_processed(self) -> None:
        with patch.object(whatsapp_routes, "_handle_image_message") as handle_image:
            handled = whatsapp_routes._handle_message(self._payload_group_image())

        self.assertTrue(handled)
        handle_image.assert_called_once()
        _, sender, reply_target, _ = handle_image.call_args.args
        self.assertEqual(sender, "5511888888888")
        self.assertEqual(reply_target, "120363025481-12345@g.us")
        self.assertEqual(self.rate_store.recorded, [("5511888888888", "image")])

    def test_group_text_with_mention_processed(self) -> None:
        with patch("app.chatbot.gerar_resposta", return_value="resposta") as gerar_resposta:
            handled = whatsapp_routes._handle_message(self._payload_group(with_mention=True))

        self.assertTrue(handled)
        gerar_resposta.assert_called_once()
        self.assertEqual(gerar_resposta.call_args.kwargs["sender"], "5511888888888")
        self.client.send_text.assert_called_once_with("120363025481-12345@g.us", "resposta")
        self.assertEqual(self.rate_store.recorded, [("5511888888888", "text")])

    def test_group_sender_uses_participant_not_group_jid(self) -> None:
        with patch("app.chatbot.gerar_resposta", return_value="resposta") as gerar_resposta:
            handled = whatsapp_routes._handle_message(
                self._payload_group(msg_id="GRP2", text="como regar?", with_mention=True)
            )

        self.assertTrue(handled)
        self.assertEqual(gerar_resposta.call_args.kwargs["sender"], "5511888888888")
        self.assertNotEqual(gerar_resposta.call_args.kwargs["sender"], "120363025481-12345")

    def test_allowed_groups_filter(self) -> None:
        with patch.dict(os.environ, {"ALLOWED_GROUPS": "120363999999-00000@g.us"}, clear=False):
            with patch("app.chatbot.gerar_resposta", return_value="resposta") as gerar_resposta:
                handled = whatsapp_routes._handle_message(
                    self._payload_group(msg_id="GRP3", with_mention=True)
                )

        self.assertFalse(handled)
        gerar_resposta.assert_not_called()
        self.client.send_text.assert_not_called()
        self.assertEqual(self.rate_store.recorded, [])

    def test_rate_limit_blocks_after_5_images(self) -> None:
        self.rate_store.counts[("5511888888888", "image")] = 5
        with patch.object(whatsapp_routes, "_handle_image_message") as handle_image:
            handled = whatsapp_routes._handle_message(self._payload_group_image(msg_id="IMG2"))

        self.assertFalse(handled)
        handle_image.assert_not_called()
        self.client.send_text.assert_called_once()
        args = self.client.send_text.call_args.args
        self.assertEqual(args[0], "120363025481-12345@g.us")
        self.assertIn("limite de 5 fotos/hora", args[1])
        self.assertEqual(self.rate_store.recorded, [])


if __name__ == "__main__":
    unittest.main()
