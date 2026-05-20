"""Chatbot especializado em germinação — resposta baseada em contexto + keywords."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from app.database import get_history, get_temporal_series

# ── Base de conhecimento ───────────────────────────────────────────────────────

CLASSES_INFO = {
    "Germinacao": {
        "nome": "Germinação (muda detectada)",
        "desc": "Indica uma muda que germinou com sucesso e está visível na bandeja. O modelo detecta a planta inteira (caule + folhas agrupadas), funciona para diversas culturas (morango, hortaliças, sementes em geral).",
        "acao": "Continue o monitoramento. Garanta rega adequada e iluminação uniforme. Compare com a capacidade total da bandeja para estimar a taxa de germinação.",
        "cor": "#34d399",
    },
    "Folha": {
        "nome": "Folha individual",
        "desc": "Cada folha detectada individualmente. Comparada com o número de germinações, indica o estágio fenológico (mais folhas = muda mais desenvolvida).",
        "acao": "Use a razão folhas/germinacao como métrica de vigor. Mudas com 3-4 folhas verdadeiras geralmente estão prontas para transplante (especialmente hortaliças e morango).",
        "cor": "#fbbf24",
    },
}

DICAS_GERMINACAO = [
    "💧 Mantenha o substrato úmido mas sem encharcamento — o excesso de água causa podridão de raiz em qualquer cultura.",
    "🌡️ A temperatura ideal para germinação fica entre 18°C e 25°C (hortaliças). Morango prefere a faixa mais baixa, 18–22°C.",
    "💡 Após a emergência, as mudas precisam de 12–16h de luz por dia para crescimento saudável.",
    "🌱 Bandejas de 128 ou 200 células são padrão comercial para produção de mudas.",
    "🧪 O pH ideal do substrato fica entre 5.5 e 6.5 para a maioria das culturas (hortaliças e morango).",
    "⏱️ Tempo de germinação varia: alface 3–7 dias, tomate 5–10 dias, manjericão 7–14 dias, morango 14–28 dias.",
    "🔄 Rotacione as bandejas periodicamente para garantir desenvolvimento uniforme entre as células.",
    "📊 Taxas de germinação: >80% é excelente (hortaliças), >70% é bom para morango (sementes têm dormência natural).",
    "🌿 Mudas com 3–4 folhas verdadeiras geralmente estão prontas para transplante.",
    "🍓 Sementes de morango se beneficiam de pré-frio (estratificação) de 2–4 semanas para quebrar dormência.",
    "🌾 Para sementes pequenas (alface, manjericão), não cubra com muito substrato — luz pode ser necessária para germinar.",
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

from app.database import get_chat_history, insert_chat_message
from app.embeddings import find_similar_messages, find_relevant_knowledge, find_similar_analyses


# ── Sanitização de formatação WhatsApp ─────────────────────────────────────────

def _sanitize_whatsapp(text: str) -> str:
    """Normaliza Markdown para o subset que o WhatsApp realmente renderiza.

    Regras corrigidas:
    - **bold** ou ***bold*** -> *bold*  (WhatsApp só aceita asterisco simples)
    - __italic__ -> _italic_
    - ~~strike~~ -> ~strike~
    - Cabeçalhos Markdown (# ## ###) -> *texto* na linha
    - Blockquotes (> ) removidos
    - Asteriscos colados com pontuação no fim do delimitador, ex "*texto* :" -> "*texto*:"
    - Trim em espaços internos do delimitador, ex "* texto *" -> "*texto*"
    """
    if not text:
        return text

    # Cabeçalhos Markdown -> negrito simples na linha
    text = re.sub(r"^\s*#{1,6}\s+(.+?)\s*$", r"*\1*", text, flags=re.MULTILINE)

    # Blockquotes -> linha normal
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)

    # Negrito Markdown duplo/triplo -> simples (***x*** primeiro pra não sobrar resíduo)
    text = re.sub(r"\*{3,}([^*\n]+?)\*{3,}", r"*\1*", text)
    text = re.sub(r"\*{2}([^*\n]+?)\*{2}", r"*\1*", text)

    # Itálico Markdown __x__ -> _x_
    text = re.sub(r"__([^_\n]+?)__", r"_\1_", text)

    # Tachado Markdown ~~x~~ -> ~x~
    text = re.sub(r"~~([^~\n]+?)~~", r"~\1~", text)

    # Remove espaços internos próximos aos delimitadores: "* texto *" -> "*texto*"
    text = re.sub(r"\*\s+([^*\n]+?)\s+\*", r"*\1*", text)
    text = re.sub(r"\*\s+([^*\n]+?)\*", r"*\1*", text)
    text = re.sub(r"\*([^*\n]+?)\s+\*", r"*\1*", text)

    return text


# ── Motor de resposta ──────────────────────────────────────────────────────────

def gerar_resposta(mensagem: str, db_path: str, sender: str | None = None) -> str:
    ctx = _contexto_db(db_path)
    
    # Try to load .env manually (fallback se dotenv não estiver instalado)
    try:
        with open(".env") as f:
            for line in f:
                if line.startswith("OPENROUTER_API_KEY="):
                    os.environ["OPENROUTER_API_KEY"] = line.strip().split("=", 1)[1]
    except Exception:
        pass

    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key or api_key == "your_openrouter_api_key_here":
        return (
            "⚠️ **Inteligência Artificial Pausada**\n\n"
            "O GerminaBot está configurado para usar a IA gratuita via OpenRouter!\n\n"
            "Abra o arquivo `.env` na pasta do projeto e adicione a sua chave:\n"
            "`OPENROUTER_API_KEY=sk-or-v1-...`\n\n"
            "👉 *Você pode criar uma chave gratuita em openrouter.ai*"
        )

    # Contexto rico para o modelo
    total = ctx.get("total", 0)
    avg_germ = ctx.get("avg_germ", 0)
    avg_leaf = ctx.get("avg_leaf", 0)
    last_germ = ctx.get("last_germ")
    days = ctx.get("days_tracked", 0)
    last_germ_txt = f"{last_germ}%" if last_germ is not None else "sem análise ainda"

    system_prompt = f"""Você é o **GerminaBot**, assistente agrícola do projeto GerminaVision — sistema de visão computacional (YOLO11) que analisa bandejas de mudas e detecta:
• **Germinacao** — muda germinada (caule + folhas agrupadas)
• **Folha** — folhas individuais (indicador de vigor/estágio)

O modelo foi treinado em dataset misto: imagens públicas de mudas em bandejas (Roboflow — diversas hortaliças, plântulas variadas, ~768 imagens) combinadas com imagens próprias de *morango* (~512 tiles em alta resolução). Você atende cultivos variados, com conhecimento específico tanto de hortaliças quanto de morango.

SUAS CAPACIDADES (o que VOCÊ — o assistente — pode fazer):
• *Análise de imagens DIRETO no WhatsApp* — quando o usuário envia foto de bandeja, você (via webhook) processa automaticamente com YOLO11, envia imagem anotada com bboxes de detecção e métricas (taxa de germinação, contagem, folhas por planta).
• *Análise de imagens no dashboard web* — em http://localhost:5001, o usuário pode fazer upload via drag-and-drop.
• Conversa com memória persistente (lembra de conversas anteriores via SQLite + busca semântica).
• Comandos rápidos no WhatsApp: digite *status*, *dica*, *histórico* ou *ajuda* para respostas estruturadas.
• Consulta de análises passadas via memória semântica.

COMO O USUÁRIO USA ANÁLISE DE IMAGEM:
• WhatsApp: anexa foto da bandeja e envia. Você responde com resultado em ~3-8 segundos.
• Web: arrasta a imagem para a página inicial e clica Analisar.

CONTEXTO DO USUÁRIO (dados reais do banco):
• Imagens analisadas: {total}
• Taxa média de germinação: {avg_germ}% (sobre bandejas de 200 células)
• Média de folhas por planta: {avg_leaf}
• Última análise: {last_germ_txt}
• Dias rastreados: {days}

CONHECIMENTO MULTI-CULTURA:
• **Hortaliças** (alface, tomate, manjericão, brássicas): germinação 3–14 dias, temp 18–25°C, taxa boa ≥80%.
• **Morango** (Fragaria × ananassa): germinação 14–28 dias (dormência natural, pré-frio ajuda), temp 18–22°C, taxa boa ≥70%.
• Comum a ambos: pH substrato 5.5–6.5, bandeja 128/200 células padrão, 3–4 folhas verdadeiras = transplante.

GLOSSÁRIO (responda direto, sem pedir esclarecimento):
• **seedling / muda** = planta jovem recém-germinada (1–4 folhas pequenas).
• **germinação** = brotamento da semente. No GerminaVision, é a classe "Germinacao".
• **estiolamento** = caule fino e alongado por falta de luz.
• **transplante** = mover muda da bandeja para vaso/canteiro definitivo.
• **dormência** = mecanismo natural que impede germinação imediata (típico em morango, framboesa).

CLASSES ORIGINAIS DO DATASET ROBOFLOW (todas hoje unificadas como "Germinacao"):
• *seedling* = muda saudável (planta bem estabelecida, crescimento normal). É o caso ideal.
• *twoseedling* = duas mudas na mesma cavidade (excesso de sementes — pode requerer desbaste).
• *weak* = muda fraca (caule fino, poucas folhas, coloração amarelada — sintoma de estiolamento, deficiência de N/Fe ou substrato pobre).
• *askew* = muda inclinada (cresceu torta por fototropismo ou substrato irregular).
• *processed* = cavidade já processada/colhida.
• *noseedling* = sem germinação detectável (falha de semente, substrato inadequado, problema hídrico).
Quando o usuário perguntar sobre qualquer um desses termos, explique DIRETO usando essas definições — NÃO redirecione como off-topic.

FORMATAÇÃO WHATSAPP (CRÍTICO — não use Markdown padrão):
• Negrito: *texto* (UM asterisco, NUNCA dois)
• Itálico: _texto_ (UM underscore)
• Tachado: ~texto~
• Monoespaçado: ```código``` (3 backticks)
• NUNCA use **texto** (asterisco duplo) — aparece como literais no WhatsApp
• NUNCA use # ## ou > (cabeçalhos/quotes não renderizam no WhatsApp)
• Listas: use • ou - no início da linha

REGRAS DE RESPOSTA (CRÍTICO):
1. Responda direto, completo, em 2-4 frases. NUNCA corte no meio.
2. Se o usuário não disser qual cultura, responda de forma geral; se mencionar morango ou hortaliça, dê informação específica daquela cultura.
3. Mantenha contexto da conversa anterior (você tem o histórico). "Como assim?" = elabore a resposta anterior, NÃO peça esclarecimento.
4. Use 1-2 emojis no máximo (🌱 🍃 🌿 💧 🌡️ 📊 🍓 quando falar de morango).
5. Use *negrito* (asterisco simples) em métricas e termos-chave. Listas só se >3 itens.
6. SÓ peça esclarecimento se a pergunta for genuinamente off-topic ou totalmente vaga (ex: "oi").
7. SÓ redirecione como off-topic se a pergunta for claramente fora de agricultura (ex: "qual a capital do Japão?"). Termos técnicos em inglês relacionados a mudas, plantas, classes do modelo, métricas — SEMPRE explique direto.

8. NUNCA invente capacidades, canais, apps ou serviços que não estão listados em SUAS CAPACIDADES. NÃO existe app móvel separado, NÃO existe email de suporte. As únicas duas formas de análise são: (1) foto direto no WhatsApp, (2) upload no dashboard web localhost:5001.
9. Se perguntarem "você consegue analisar imagem?" — responda SIM, e oriente: "Pode mandar a foto aqui mesmo no WhatsApp e eu processo automaticamente."

INTEGRIDADE DA RESPOSTA (CRÍTICO — revise antes de enviar):
• Nunca omita negações: "não", "sem", "evite", "jamais" devem aparecer exatamente como são.
• Frases completas: não truncar palavras-chave nem ideias no meio da sentença.
• Coerência factual: NÃO inventar dados ou parâmetros sobre culturas — use apenas o conhecimento do prompt.
"""

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost:5001",
        "X-Title": "GerminaVision",
    }

    # Modelos free (catálogo OpenRouter 2026-05). Vários fallbacks porque free tier sofre 429.
    # Modelos pequenos ficam menos saturados (rate-limit upstream menos agressivo).
    models = [
        "meta-llama/llama-3.3-70b-instruct:free",
        "z-ai/glm-4.5-air:free",
        "openai/gpt-oss-20b:free",
        "nvidia/nemotron-nano-9b-v2:free",
        "google/gemma-4-26b-a4b-it:free",
        "qwen/qwen3-next-80b-a3b-instruct:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "openai/gpt-oss-120b:free",
    ]

    # Bloco 1: RAG sobre knowledge_base agronômico
    knowledge_block = ""
    try:
        relevant_chunks = find_relevant_knowledge(db_path, mensagem, top_k=3, threshold=0.3)
        if relevant_chunks:
            lines = [f"• {item['chunk'][:300]}" for item in relevant_chunks]
            knowledge_block = "\n\nCONHECIMENTO RELEVANTE (do material do projeto, use como fonte primária):\n" + "\n".join(lines)
    except Exception:
        pass

    # Bloco 2: análises históricas similares
    analyses_block = ""
    try:
        similar_analyses = find_similar_analyses(db_path, mensagem, top_k=2, threshold=0.25)
        if similar_analyses:
            lines = [f"• {item['summary']}" for item in similar_analyses]
            analyses_block = "\n\nANÁLISES SIMILARES NO HISTÓRICO:\n" + "\n".join(lines)
    except Exception:
        pass

    # Bloco 3: memória semântica de mensagens antigas
    semantic_block = ""
    if sender:
        try:
            similar = find_similar_messages(db_path, sender, mensagem, top_k=5, exclude_recent=20)
            if similar:
                lines = [f"• [{msg['role']}] {msg['content'][:200]}" for msg in similar]
                semantic_block = "\n\nMEMÓRIA RELEVANTE (mensagens antigas com contexto útil):\n" + "\n".join(lines)
        except Exception:
            pass

    # Monta mensagens: system + histórico cronológico + mensagem atual
    history = get_chat_history(db_path, sender, limit=20) if sender else []
    full_system = system_prompt + knowledge_block + analyses_block + semantic_block
    messages = [{"role": "system", "content": full_system}]
    messages.extend(history)
    messages.append({"role": "user", "content": mensagem})

    last_err = ""
    for model in models:
        data = {
            "model": model,
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 600,
            "top_p": 0.9,
        }
        req = urllib.request.Request(
            url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                result = json.loads(response.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"].strip()
                if content:
                    content = _sanitize_whatsapp(content)
                    if sender:
                        insert_chat_message(db_path, sender, "user", mensagem)
                        insert_chat_message(db_path, sender, "assistant", content)
                    return content
                last_err = "resposta vazia"
        except urllib.error.HTTPError as e:
            last_err = f"{model}: HTTP {e.code}"
            continue
        except urllib.error.URLError as e:
            last_err = f"{model}: {e}"
            continue
        except Exception as e:
            last_err = f"{model}: {e}"
            continue

    return f"🚨 Todos os modelos falharam. Último erro: {last_err}"
