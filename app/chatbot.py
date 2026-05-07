"""Chatbot especializado em germinação — resposta baseada em contexto + keywords."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from app.database import get_history, get_temporal_series

# ── Base de conhecimento ───────────────────────────────────────────────────────

CLASSES_INFO = {
    "seedling": {
        "nome": "Muda saudável (seedling)",
        "desc": "Uma muda bem estabelecida, com crescimento normal. É a classe desejada — indica germinação bem-sucedida e desenvolvimento adequado.",
        "acao": "Continue o monitoramento. Garanta rega adequada e iluminação uniforme.",
        "cor": "#34d399",
    },
    "twoseedling": {
        "nome": "Duas mudas (twoseedling)",
        "desc": "Duas mudas emergiram na mesma cavidade. Pode indicar excesso de sementes por vaso.",
        "acao": "Considere desbaste (remoção de uma das mudas) para evitar competição por nutrientes.",
        "cor": "#10b981",
    },
    "weak": {
        "nome": "Muda fraca (weak)",
        "desc": "A muda germinou mas apresenta crescimento débil — caule fino, poucas folhas ou coloração amarelada.",
        "acao": "Verifique iluminação (estiolamento), pH do substrato e disponibilidade de nutrientes (N, Fe).",
        "cor": "#fbbf24",
    },
    "noseedling": {
        "nome": "Sem germinação (noseedling)",
        "desc": "A cavidade não apresentou germinação detectável. Pode ser falha de semente, substrato inadequado ou problema hídrico.",
        "acao": "Reavalie a qualidade das sementes, temperatura e umidade do substrato. Replantio pode ser necessário.",
        "cor": "#ef4444",
    },
    "processed": {
        "nome": "Processada (processed)",
        "desc": "Cavidade que já foi processada/colhida ou está em etapa avançada do ciclo.",
        "acao": "Registre para fins de rastreabilidade do lote.",
        "cor": "#8b5cf6",
    },
    "askew": {
        "nome": "Inclinada (askew)",
        "desc": "A muda germinou mas cresceu de forma inclinada, possivelmente por fototropismo (luz unidirecional) ou substrato irregular.",
        "acao": "Corrija a fonte de luz para iluminação uniforme. Verifique se o substrato está compactado de forma desigual.",
        "cor": "#f97316",
    },
}

DICAS_GERMINACAO = [
    "💧 Mantenha o substrato úmido mas sem encharcamento — o excesso de água causa podridão de raiz.",
    "🌡️ A temperatura ideal para germinação da maioria das hortaliças fica entre 18°C e 25°C.",
    "💡 Após a emergência, as mudas precisam de 12–16h de luz por dia para crescimento saudável.",
    "🌱 Bandejas de isopor com 128 ou 200 células são ideais para produção de mudas em escala.",
    "🧪 O pH ideal do substrato para hortaliças está entre 5.5 e 6.5.",
    "⏱️ Sementes de alface germinam em 3–7 dias; tomate em 5–10 dias; manjericão em 7–14 dias.",
    "🔄 Rotacione as bandejas periodicamente para garantir desenvolvimento uniforme das mudas.",
    "📊 Uma taxa de germinação acima de 80% é considerada excelente para produção comercial.",
]

SAUDACOES = ["olá", "oi", "hello", "hey", "bom dia", "boa tarde", "boa noite", "ei"]
DESPEDIDAS = ["tchau", "até", "obrigado", "valeu", "falou", "bye"]

# ── Helpers ────────────────────────────────────────────────────────────────────

def _hora_do_dia() -> str:
    h = datetime.now().hour
    if h < 12: return "Bom dia"
    if h < 18: return "Boa tarde"
    return "Boa noite"


def _contexto_db(db_path: str) -> dict:
    """Carrega estatísticas do histórico para respostas contextuais."""
    try:
        history  = get_history(db_path, limit=100)
        temporal = get_temporal_series(db_path)
        if not history:
            return {"total": 0}
        avg_germ = sum(r["germination_rate"] for r in history) / len(history)
        avg_leaf = sum(r["leaf_avg"] for r in history) / len(history)
        return {
            "total":        len(history),
            "avg_germ":     round(avg_germ, 1),
            "avg_leaf":     round(avg_leaf, 1),
            "last_germ":    history[0]["germination_rate"] if history else None,
            "days_tracked": len(temporal),
        }
    except Exception:
        return {"total": 0}


import os
import json
import urllib.request
import urllib.error

# ── Motor de resposta ──────────────────────────────────────────────────────────

def gerar_resposta(mensagem: str, db_path: str) -> str:
    ctx = _contexto_db(db_path)
    
    # Try to load .env manually (fallback se dotenv não estiver instalado)
    try:
        with open(".env") as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY="):
                    os.environ["OPENAI_API_KEY"] = line.strip().split("=", 1)[1]
    except Exception:
        pass

    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key or api_key == "your_openai_api_key_here":
        return (
            "⚠️ **Inteligência Artificial Pausada**\n\n"
            "O GerminaBot agora usa IA Generativa real! Para funcionar, você precisa configurar a sua **API Key da OpenAI**.\n\n"
            "Abra o arquivo `.env` na pasta do projeto e adicione a sua chave:\n"
            "`OPENAI_API_KEY=sk-...`\n\n"
            "👉 *Dica: você também pode conferir a aba **MCP API** no menu para integrar seu projeto com o Claude Desktop ou Cursor!*"
        )

    system_prompt = (
        "Você é o GerminaBot, um assistente agrícola inteligente do projeto GerminaVision.\n"
        "O GerminaVision usa Visão Computacional (YOLO11) para detectar germinação de sementes e estimar a contagem de folhas.\n"
        "Seja cordial, use emojis, e dê respostas diretas baseadas nas estatísticas reais do usuário fornecidas abaixo.\n\n"
        "--- ESTATÍSTICAS ATUAIS DO USUÁRIO ---\n"
        f"- Total de imagens analisadas: {ctx.get('total', 0)}\n"
        f"- Taxa média de germinação: {ctx.get('avg_germ', 0)}%\n"
        f"- Média de folhas por muda: {ctx.get('avg_leaf', 0)}\n"
        f"- Última taxa de germinação: {ctx.get('last_germ', 'N/A')}%\n"
        f"- Dias rastreados na evolução temporal: {ctx.get('days_tracked', 0)}\n"
        "----------------------------------------\n\n"
        "Regras para perguntas sobre viabilidade/produção:\n"
        "- >= 80% germinação: Excelente, retorno comercial garantido.\n"
        "- 60% a 79%: Moderado. Viável, mas com perda de eficiência nos espaços vazios da bandeja.\n"
        "- Abaixo de 60%: Risco de prejuízo. A mão de obra supera o lucro.\n\n"
        "Use formatação Markdown amigável."
    )

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": mensagem}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }

    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
    except urllib.error.URLError as e:
        return f"🚨 Erro de conexão com a API da OpenAI: {e}"
    except Exception as e:
        return f"🚨 Erro interno no bot: {e}"
