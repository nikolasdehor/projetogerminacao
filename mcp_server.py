#!/usr/bin/env python3
"""
GerminaVision MCP Server

Servidor MCP que expõe as funcionalidades do GerminaVision para qualquer
agente de IA compatível com o Model Context Protocol (Claude, Cursor, etc.).

Uso:
    python mcp_server.py

Configuração no claude_desktop_config.json:
    {
      "mcpServers": {
        "germinavision": {
          "command": "<caminho-para-venv>/bin/python",
          "args": ["<caminho-para-projeto>/mcp_server.py"]
        }
      }
    }
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import httpx
from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP

# ── Config ─────────────────────────────────────────────────────────────────────
FLASK_BASE = "http://localhost:5001"
TIMEOUT    = 60.0   # segundos (análise de imagem pode demorar)

mcp = FastMCP("germinavision_mcp")

# ── Shared helpers ─────────────────────────────────────────────────────────────

async def _get(endpoint: str) -> dict | list:
    async with httpx.AsyncClient(base_url=FLASK_BASE, timeout=TIMEOUT) as client:
        r = await client.get(endpoint)
        r.raise_for_status()
        return r.json()


def _err(e: Exception) -> str:
    if isinstance(e, httpx.ConnectError):
        return "Erro: GerminaVision não está rodando. Execute `python run.py` no diretório do projeto."
    if isinstance(e, httpx.HTTPStatusError):
        return f"Erro HTTP {e.response.status_code}: {e.response.text[:200]}"
    if isinstance(e, httpx.TimeoutException):
        return "Erro: tempo limite esgotado. A análise demorou mais do que o esperado."
    return f"Erro inesperado: {type(e).__name__}: {e}"


# ── Input models ───────────────────────────────────────────────────────────────

class AnalyzeInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    image_path: str = Field(
        ...,
        description="Caminho absoluto para a imagem a analisar (ex: '/Users/user/foto.jpg'). "
                    "Aceita PNG, JPG, WEBP, BMP.",
    )
    confidence: float = Field(
        default=0.25,
        description="Confiança mínima para detectar uma muda (0.10 a 0.90). Padrão: 0.25.",
        ge=0.10, le=0.90,
    )
    day_label: Optional[str] = Field(
        default=None,
        description="Rótulo do dia para série temporal (ex: 'D0', 'D1', 'D3'). Opcional.",
        max_length=20,
    )


class HistoryInput(BaseModel):
    limit: int = Field(default=20, description="Número máximo de registros a retornar (1-100).", ge=1, le=100)


class ChatInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    message: str = Field(..., description="Pergunta ou mensagem para o GerminaBot.", min_length=1, max_length=300)


# ── Tools ──────────────────────────────────────────────────────────────────────

@mcp.tool(
    name="germina_analyze_image",
    annotations={
        "title": "Analisar Imagem de Mudas",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def germina_analyze_image(params: AnalyzeInput) -> str:
    """Analisa uma imagem de bandeja de mudas usando o modelo YOLO11 treinado.

    Detecta automaticamente mudas nas categorias: seedling (saudável),
    twoseedling (dupla), weak (fraca), noseedling (sem germinação),
    processed (processada) e askew (inclinada).

    Retorna taxa de germinação, número de mudas detectadas, estimativa de
    folhas por muda, tempo de inferência e caminho para imagem anotada.

    Args:
        params (AnalyzeInput): Parâmetros contendo:
            - image_path (str): Caminho absoluto da imagem
            - confidence (float): Limiar de confiança (padrão 0.25)
            - day_label (str, opcional): Rótulo do dia (D0, D1…)

    Returns:
        str: JSON com campos: germination_rate, total_detected, germinated,
             leaf_avg, inference_time_s, result_image, detections[], id
    """
    path = Path(params.image_path)
    if not path.exists():
        return f"Erro: arquivo não encontrado: {params.image_path}"
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        return f"Erro: formato não suportado '{path.suffix}'. Use JPG, PNG, WEBP ou BMP."

    try:
        async with httpx.AsyncClient(base_url=FLASK_BASE, timeout=TIMEOUT) as client:
            with open(path, "rb") as f:
                files   = {"image": (path.name, f, "image/jpeg")}
                data    = {"conf": str(params.confidence)}
                if params.day_label:
                    data["day_label"] = params.day_label
                r = await client.post("/api/analyze", files=files, data=data)
            r.raise_for_status()
            result = r.json()

        # Formato legível para o agente
        detections_summary = "\n".join(
            f"  • {d['class']} (conf {d['confidence']:.0%}, {d['leaf_count']} folhas)"
            for d in result.get("detections", [])
        )
        return (
            f"## Resultado da análise\n\n"
            f"- **Taxa de germinação:** {result['germination_rate']}%\n"
            f"- **Total detectadas:** {result['total_detected']}\n"
            f"- **Germinadas:** {result['germinated']}\n"
            f"- **Folhas médias/muda:** {result['leaf_avg']}\n"
            f"- **Tempo de inferência:** {result['inference_time_s']}s\n"
            f"- **ID do registro:** {result.get('id')}\n"
            f"- **Imagem anotada:** {FLASK_BASE}{result['result_image']}\n\n"
            f"### Detecções\n{detections_summary or 'Nenhuma detecção.'}\n\n"
            f"```json\n{json.dumps(result, indent=2, ensure_ascii=False)}\n```"
        )
    except Exception as e:
        return _err(e)


@mcp.tool(
    name="germina_get_history",
    annotations={
        "title": "Buscar Histórico de Análises",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def germina_get_history(params: HistoryInput) -> str:
    """Retorna o histórico das últimas análises realizadas no GerminaVision.

    Cada registro inclui: id, timestamp, arquivo, taxa de germinação,
    mudas detectadas, mudas germinadas, folhas médias e rótulo do dia.

    Args:
        params (HistoryInput): limit — número de registros (padrão 20, máx 100)

    Returns:
        str: Tabela markdown com os registros mais recentes e JSON completo.
    """
    try:
        records = await _get(f"/api/history?limit={params.limit}")
        if not records:
            return "Nenhuma análise registrada ainda. Use `germina_analyze_image` para começar."

        rows = "\n".join(
            f"| {r['id']} | {r['timestamp']} | {r.get('day_label') or '—'} "
            f"| {r['germination_rate']}% | {r['total_detected']} | {r['germinated']} | {r['leaf_avg']} |"
            for r in records
        )
        table = (
            "| # | Data/Hora | Dia | Taxa | Detectadas | Germinadas | Folhas |\n"
            "|---|-----------|-----|------|------------|------------|--------|\n"
            f"{rows}"
        )
        return f"## Histórico ({len(records)} registros)\n\n{table}"
    except Exception as e:
        return _err(e)


@mcp.tool(
    name="germina_get_stats",
    annotations={
        "title": "Estatísticas Gerais do Experimento",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def germina_get_stats() -> str:
    """Retorna estatísticas agregadas de todas as análises do experimento.

    Inclui: total de análises, taxa média/máxima/mínima de germinação,
    folhas médias, total de mudas detectadas/germinadas, dias rastreados
    e distribuição de qualidade (Ótima/Boa/Regular/Baixa).

    Returns:
        str: Relatório markdown com métricas globais e distribuição.
    """
    try:
        data    = await _get("/api/stats")
        summary = data.get("summary", {})
        dist    = data.get("distribution", [])

        if not summary.get("total"):
            return "Nenhuma análise registrada ainda."

        dist_lines = "\n".join(f"  - {d['faixa']}: {d['qtd']} análise(s)" for d in dist)
        return (
            f"## Estatísticas do Experimento\n\n"
            f"- **Total de análises:** {summary['total']}\n"
            f"- **Taxa média de germinação:** {summary['avg_rate']}%\n"
            f"- **Melhor taxa:** {summary['best_rate']}%\n"
            f"- **Pior taxa:** {summary['worst_rate']}%\n"
            f"- **Folhas médias/muda:** {summary['avg_leaves']}\n"
            f"- **Total de mudas detectadas:** {summary['total_detected']}\n"
            f"- **Total de mudas germinadas:** {summary['total_germinated']}\n"
            f"- **Dias rastreados:** {summary['days_tracked']}\n\n"
            f"### Distribuição de qualidade\n{dist_lines}"
        )
    except Exception as e:
        return _err(e)


@mcp.tool(
    name="germina_get_temporal",
    annotations={
        "title": "Série Temporal de Germinação",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def germina_get_temporal() -> str:
    """Retorna a série temporal de germinação agrupada por rótulo de dia.

    Mostra a evolução da taxa de germinação e folhas médias ao longo dos
    dias do experimento (D0, D1, D2…). Requer que as análises tenham sido
    feitas com o campo 'day_label' preenchido.

    Returns:
        str: Tabela markdown com a evolução temporal e JSON completo.
    """
    try:
        series = await _get("/api/temporal")
        if not series:
            return (
                "Série temporal vazia. Para ver a evolução, faça uploads com o campo "
                "'Rótulo do dia' preenchido (D0, D1, D2…) ou use o parâmetro `day_label` "
                "em `germina_analyze_image`."
            )

        rows = "\n".join(
            f"| {p['day']} | {p['avg_germination_rate']:.1f}% | {p['avg_leaf_count']:.1f} | {p['num_analyses']} |"
            for p in series
        )
        return (
            f"## Evolução Temporal ({len(series)} pontos)\n\n"
            "| Dia | Taxa Germinação | Folhas Médias | Análises |\n"
            "|-----|-----------------|---------------|----------|\n"
            f"{rows}"
        )
    except Exception as e:
        return _err(e)


@mcp.tool(
    name="germina_ask",
    annotations={
        "title": "Perguntar ao GerminaBot",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def germina_ask(params: ChatInput) -> str:
    """Faz uma pergunta ao GerminaBot, especialista em germinação e cultivo de mudas.

    O GerminaBot tem conhecimento sobre as classes do modelo YOLO, técnicas de
    cultivo, interpretação de métricas e histórico de análises do experimento.

    Args:
        params (ChatInput): message — pergunta em linguagem natural (pt-BR)

    Returns:
        str: Resposta do GerminaBot com informações sobre germinação.

    Exemplos de perguntas:
        - "O que significa a classe weak?"
        - "Qual a minha taxa de germinação?"
        - "Dicas para melhorar a germinação"
        - "Como funciona o modelo YOLO?"
    """
    try:
        async with httpx.AsyncClient(base_url=FLASK_BASE, timeout=30.0) as client:
            r = await client.post(
                "/api/chat",
                json={"message": params.message},
                headers={"Content-Type": "application/json"},
            )
            r.raise_for_status()
            data = r.json()
        return data.get("reply") or data.get("error") or "Sem resposta."
    except Exception as e:
        return _err(e)


@mcp.tool(
    name="germina_status",
    annotations={
        "title": "Status do Servidor GerminaVision",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def germina_status() -> str:
    """Verifica se o servidor GerminaVision está online e qual modelo está carregado.

    Returns:
        str: Status do servidor, modelo carregado e URL de acesso.
    """
    try:
        data = await _get("/api/status")
        modelo = "✅ Modelo personalizado (best.pt)" if data.get("custom_model") else "⚠️ Modelo COCO pré-treinado (sem best.pt)"
        return (
            f"## Status do GerminaVision\n\n"
            f"- **Servidor:** {'✅ Online' if data.get('model_loaded') else '❌ Offline'}\n"
            f"- **Modelo:** {modelo}\n"
            f"- **URL:** {FLASK_BASE}\n"
            f"- **Versão:** {data.get('version', 'N/A')}"
        )
    except Exception as e:
        return _err(e)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🌱 GerminaVision MCP Server iniciando…", flush=True)
    mcp.run()
