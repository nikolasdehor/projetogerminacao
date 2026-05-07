#!/bin/bash
# ============================================================
# setup.sh — Configura o ambiente virtual e instala dependências
# Uso: bash setup.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "🌱  GerminaVision — Setup"
echo "================================"

# Python do Homebrew ou sistema
if [ -f "/opt/homebrew/bin/python3" ]; then
    PYTHON="/opt/homebrew/bin/python3"
elif command -v python3 &>/dev/null; then
    PYTHON="$(which python3)"
else
    echo "❌  Python 3 não encontrado. Instale com: brew install python3"
    exit 1
fi

echo "🐍  Python: $($PYTHON --version)"

# Cria venv se não existir
if [ ! -d "venv" ]; then
    echo "📦  Criando ambiente virtual em ./venv …"
    $PYTHON -m venv venv
fi

# Ativa e instala dependências
echo "⬇️   Instalando dependências …"
./venv/bin/pip install --upgrade pip -q
./venv/bin/pip install flask ultralytics opencv-python-headless Pillow numpy

echo ""
echo "✅  Setup concluído!"
echo ""
echo "Para iniciar a aplicação:"
echo "  ./venv/bin/python run.py"
echo ""
echo "Ou ative o venv e rode:"
echo "  source venv/bin/activate"
echo "  python run.py"
echo ""
