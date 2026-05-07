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


def _normalizar(texto: str) -> str:
    return re.sub(r"[^a-z0-9 áéíóúãõâêîôûàç]", " ", texto.lower().strip())


# ── Motor de resposta ──────────────────────────────────────────────────────────

def gerar_resposta(mensagem: str, db_path: str) -> str:
    msg   = _normalizar(mensagem)
    ctx   = _contexto_db(db_path)
    words = set(msg.split())

    # ── Saudação ──────────────────────────────────────────────────────────────
    if any(s in msg for s in SAUDACOES) and len(msg.split()) <= 5:
        return (
            f"{_hora_do_dia()}! 🌱 Sou o **GerminaBot**, assistente especializado "
            f"em monitoramento de germinação e crescimento de mudas.\n\n"
            f"Posso te ajudar com:\n"
            f"- 🔍 Explicar as classes detectadas (seedling, weak, noseedling...)\n"
            f"- 📊 Analisar seus resultados\n"
            f"- 🌿 Dicas de cultivo e germinação\n"
            f"- ❓ Tirar dúvidas sobre o projeto\n\n"
            f"O que você quer saber?"
        )

    # ── Despedida ─────────────────────────────────────────────────────────────
    if any(d in msg for d in DESPEDIDAS):
        return "Até logo! 🌱 Bons cultivos e boa taxa de germinação!"

    # ── Classes do modelo ─────────────────────────────────────────────────────
    for cls, info in CLASSES_INFO.items():
        if cls in msg or cls.replace("no", "sem ") in msg:
            return (
                f"**{info['nome']}** {info['cor'] and ''}\n\n"
                f"📋 **O que é:** {info['desc']}\n\n"
                f"✅ **O que fazer:** {info['acao']}"
            )

    # ── Total / contagem geral ─────────────────────────────────────────────────
    if any(w in msg for w in ["quantas", "quanto", "total", "plantas", "analisamos", "realizamos", "registros", "fizemos", "já fiz"]):
        if ctx["total"] == 0:
            return "Ainda não há análises registradas. Faça o primeiro upload de uma imagem! 🌱"
        return (
            f"📊 **Resumo das análises:**\n\n"
            f"- Total de imagens analisadas: **{ctx['total']}**\n"
            f"- Taxa média de germinação: **{ctx['avg_germ']}%**\n"
            f"- Média de folhas por muda: **{ctx['avg_leaf']}**\n"
            + (f"- Última análise: **{ctx['last_germ']}%** de germinação\n" if ctx.get('last_germ') is not None else "")
            + f"\n{'🎉 Excelente progresso!' if ctx['avg_germ'] >= 70 else '📈 Continue monitorando para melhorar a taxa!'}"
        )

    # ── Taxa de germinação ────────────────────────────────────────────────────
    if any(w in msg for w in ["taxa", "germinação", "germinacao", "percentual", "porcentagem", "%"]):
        if ctx["total"] == 0:
            return "Você ainda não tem análises registradas. Faça o upload de uma imagem para começar!"
        resp = (
            f"📊 **Suas estatísticas de germinação:**\n\n"
            f"- Total de análises: **{ctx['total']}**\n"
            f"- Taxa média de germinação: **{ctx['avg_germ']}%**\n"
            f"- Média de folhas por muda: **{ctx['avg_leaf']}**\n"
        )
        if ctx.get("days_tracked", 0) > 1:
            resp += f"- Dias monitorados: **{ctx['days_tracked']}**\n"
        if ctx["avg_germ"] >= 80:
            resp += "\n🎉 Excelente! Taxa acima de 80% é considerada muito boa para produção comercial."
        elif ctx["avg_germ"] >= 60:
            resp += "\n👍 Taxa razoável. Verifique as cavidades com `noseedling` para identificar causas da falha."
        else:
            resp += "\n⚠️ Taxa abaixo do esperado. Revise qualidade das sementes, temperatura e umidade do substrato."
        return resp

    # ── Folhas ────────────────────────────────────────────────────────────────
    if any(w in msg for w in ["folha", "folhas", "galho", "galhos", "conta", "contagem"]):
        return (
            "🍃 **Contagem de folhas/galhos:**\n\n"
            "O sistema usa uma heurística baseada em análise de área verde para estimar o número de folhas:\n\n"
            "- **seedling**: geralmente 2–3 folhas\n"
            "- **twoseedling**: ~4–6 folhas (duas mudas juntas)\n"
            "- **weak**: 1–2 folhas (crescimento reduzido)\n"
            "- **askew**: 2–3 folhas (normal, mas inclinada)\n\n"
            "Para contagem mais precisa, você pode treinar o Modelo 2 (LeafRegressor) com anotações manuais — "
            "veja a seção correspondente no Notebook Colab do projeto."
        )

    # ── YOLO / modelo ─────────────────────────────────────────────────────────
    if any(w in msg for w in ["yolo", "modelo", "treino", "treinamento", "detecção", "deteccao", "ia", "inteligência", "rede neural"]):
        return (
            "🤖 **Sobre o modelo de detecção:**\n\n"
            "O sistema usa **YOLO11s** treinado no dataset de mudas do Roboflow com 768 imagens e 6 classes:\n\n"
            "| Classe | Significado |\n"
            "|---|---|\n"
            "| `seedling` | Muda saudável ✅ |\n"
            "| `twoseedling` | Duas mudas na mesma cavidade |\n"
            "| `weak` | Muda fraca ⚠️ |\n"
            "| `noseedling` | Sem germinação ❌ |\n"
            "| `processed` | Cavidade processada |\n"
            "| `askew` | Muda inclinada |\n\n"
            "A confiança mínima padrão é 25% — aumente para menos detecções mas mais precisas."
        )

    # ── Dica geral ────────────────────────────────────────────────────────────
    if any(w in msg for w in ["dica", "dicas", "conselho", "como", "ajuda", "help", "problema", "erro"]):
        import random
        dica = random.choice(DICAS_GERMINACAO)
        return f"🌿 **Dica de cultivo:**\n\n{dica}\n\nQuer mais dicas? É só perguntar!"

    # ── Histórico / série temporal ────────────────────────────────────────────
    if any(w in msg for w in ["histórico", "historico", "análises", "analises", "registros", "dias", "evolução", "evolucao"]):
        if ctx["total"] == 0:
            return "Você ainda não tem análises no histórico. Faça o upload de imagens e use o campo **Rótulo do dia** (D0, D1, D2...) para montar a série temporal!"
        return (
            f"📅 **Seu histórico:**\n\n"
            f"- {ctx['total']} análise(s) registrada(s)\n"
            f"- {ctx.get('days_tracked', 0)} dia(s) distintos monitorados\n"
            f"- Taxa média de germinação: {ctx['avg_germ']}%\n\n"
            f"Use o campo **Rótulo do dia** ao fazer upload (ex: D0, D1, D3) para o gráfico temporal aparecer no dashboard!"
        )

    # ── Sobre o projeto ───────────────────────────────────────────────────────
    if any(w in msg for w in ["projeto", "sistema", "app", "aplicação", "aplicacao", "flask", "python"]):
        return (
            "🌱 **Sobre o GerminaVision:**\n\n"
            "Sistema de visão computacional para monitoramento automático de germinação e crescimento de mudas.\n\n"
            "**Stack:**\n"
            "- 🐍 Python + Flask (backend)\n"
            "- 🤖 YOLO11s (detecção de mudas)\n"
            "- 🗄️ SQLite (histórico temporal)\n"
            "- 📊 Chart.js (gráficos)\n\n"
            "**Funcionalidades:**\n"
            "- Upload de imagens com detecção automática\n"
            "- Estimativa de contagem de folhas\n"
            "- Taxa de germinação por imagem\n"
            "- Dashboard temporal de evolução\n"
            "- Este chatbot 🤖"
        )

    # ── Resposta padrão ───────────────────────────────────────────────────────
    sugestoes = [
        "o que significa a classe `weak`",
        "minha taxa de germinação",
        "dicas de cultivo",
        "como funciona o modelo YOLO",
        "meu histórico de análises",
    ]
    import random
    sug = random.choice(sugestoes)
    return (
        f"Hmm, não tenho certeza sobre isso ainda. 🤔\n\n"
        f"Posso te ajudar com tópicos como:\n"
        f"- Classes detectadas (seedling, weak, noseedling...)\n"
        f"- Taxa de germinação e estatísticas\n"
        f"- Dicas de cultivo\n"
        f"- Como o modelo funciona\n\n"
        f"Tente perguntar algo como: *\"{sug}\"*"
    )
