"""Tests de recovery offline + dedupe de mensagens."""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import MagicMock

from app.processed_messages import ProcessedMessageStore


class ProcessedMessageStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db = os.path.join(self.tmpdir, "test.db")
        self.store = ProcessedMessageStore(db_path=self.db)

    def test_is_processed_false_initially(self) -> None:
        self.assertFalse(self.store.is_processed("msg-1"))

    def test_mark_then_is_processed(self) -> None:
        self.store.mark_processed("msg-1", 1700000000)
        self.assertTrue(self.store.is_processed("msg-1"))

    def test_mark_is_idempotent(self) -> None:
        self.store.mark_processed("msg-1", 1700000000)
        self.store.mark_processed("msg-1", 1700000999)  # nao falha
        self.assertTrue(self.store.is_processed("msg-1"))

    def test_latest_timestamp(self) -> None:
        self.assertIsNone(self.store.latest_timestamp())
        self.store.mark_processed("msg-1", 1700000000)
        self.store.mark_processed("msg-2", 1700001000)
        self.assertEqual(self.store.latest_timestamp(), 1700001000)

    def test_persists_between_instances(self) -> None:
        self.store.mark_processed("msg-1", 1700000000)
        new_store = ProcessedMessageStore(db_path=self.db)
        self.assertTrue(new_store.is_processed("msg-1"))


class RecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db = os.path.join(self.tmpdir, "test.db")
        self.store = ProcessedMessageStore(db_path=self.db)

        self.client = MagicMock()
        self.client.is_configured.return_value = True

    def _make_msg(self, msg_id: str, ts: int) -> dict:
        return {
            "data": {
                "key": {"id": msg_id, "fromMe": False},
                "messageTimestamp": ts,
                "message": {"imageMessage": {"url": f"http://example/{msg_id}"}},
            }
        }

    def test_processes_new_messages(self) -> None:
        from app.recovery import recover_pending_messages

        self.client.find_pending_messages.return_value = [
            self._make_msg("m1", 1700000000),
            self._make_msg("m2", 1700000010),
        ]
        processed: list[str] = []
        recover_pending_messages(
            client=self.client,
            store=self.store,
            process_fn=lambda m: processed.append(m["data"]["key"]["id"]),
        )
        self.assertEqual(processed, ["m1", "m2"])

    def test_dedupes_already_processed(self) -> None:
        from app.recovery import recover_pending_messages

        self.store.mark_processed("m1", 1700000000)
        self.client.find_pending_messages.return_value = [
            self._make_msg("m1", 1700000000),
            self._make_msg("m2", 1700000010),
        ]
        processed: list[str] = []
        recover_pending_messages(
            client=self.client,
            store=self.store,
            process_fn=lambda m: processed.append(m["data"]["key"]["id"]),
        )
        self.assertEqual(processed, ["m2"])

    def test_running_twice_no_reprocess(self) -> None:
        from app.recovery import recover_pending_messages

        msgs = [self._make_msg("m1", 1700000000)]
        self.client.find_pending_messages.return_value = msgs

        processed: list[str] = []

        recover_pending_messages(
            client=self.client,
            store=self.store,
            process_fn=lambda m: processed.append(m["data"]["key"]["id"]),
        )
        recover_pending_messages(
            client=self.client,
            store=self.store,
            process_fn=lambda m: processed.append(m["data"]["key"]["id"]),
        )
        self.assertEqual(processed, ["m1"])


if __name__ == "__main__":
    unittest.main()
