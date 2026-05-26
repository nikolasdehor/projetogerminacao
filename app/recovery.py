"""Recovery de mensagens recebidas enquanto o servico estava offline."""
from __future__ import annotations

import logging
import time
from typing import Callable

from app.whatsapp import EvolutionClient
from app.processed_messages import ProcessedMessageStore


logger = logging.getLogger(__name__)


def recover_pending_messages(
    client: EvolutionClient,
    store: ProcessedMessageStore,
    process_fn: Callable[[dict], None],
    max_age_hours: int = 24,
    limit: int = 100,
) -> int:
    """Puxa mensagens recebidas durante downtime e enfileira pra processamento.

    Args:
        client: cliente Evolution conectado.
        store: store de dedupe de mensagens.
        process_fn: funcao que processa UMA message_data (mesmo shape do webhook).
        max_age_hours: limite superior de janela de recuperacao.
        limit: numero maximo de mensagens por chamada Evolution.

    Returns:
        Numero de mensagens efetivamente recuperadas e processadas.
    """
    if not client.is_configured():
        logger.info("recovery: Evolution nao configurada, pulando")
        return 0

    last_ts = store.latest_timestamp()
    now = int(time.time())
    floor_ts = now - max_age_hours * 3600
    since = max(last_ts or floor_ts, floor_ts)

    try:
        pending = client.find_pending_messages(since_timestamp=since, limit=limit)
    except Exception as exc:
        logger.warning("recovery: falha buscando mensagens pendentes: %s", exc)
        return 0

    recovered = 0
    for msg in pending:
        message_id = (
            msg.get("data", {}).get("key", {}).get("id")
            or msg.get("data", {}).get("messageId")
            or ""
        )
        if not message_id or store.is_processed(message_id):
            continue
        try:
            process_fn(msg)
            msg_ts = msg.get("data", {}).get("messageTimestamp")
            try:
                msg_ts_int = int(msg_ts) if msg_ts is not None else None
            except (TypeError, ValueError):
                msg_ts_int = None
            store.mark_processed(message_id, msg_ts_int)
            recovered += 1
        except Exception as exc:
            logger.warning(
                "recovery: falha processando %s: %s", message_id, exc
            )

    if recovered:
        logger.info("recovery: %d mensagem(ns) recuperadas e processadas", recovered)
    return recovered
