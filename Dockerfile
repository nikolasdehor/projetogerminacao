# ── Stage 1: builder ─────────────────────────────────────────────────────────
# Instala dependências pesadas (torch CPU, ultralytics) em camada separada para
# aproveitar cache do Docker e manter a imagem final mais enxuta.
FROM python:3.10-slim AS builder

# Dependências de sistema necessárias para OpenCV headless e compilação de pacotes
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

# Instala dependências no diretório /install para copiar depois.
# torch CPU: índice oficial PyTorch CPU-only (economiza ~2 GB vs versão CUDA).
# ultralytics, flask e demais libs vêm do PyPI normalmente.
RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir \
        --target /install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.3.1+cpu \
        torchvision==0.18.1+cpu && \
    pip install --no-cache-dir \
        --target /install \
        flask>=3.0 \
        ultralytics>=8.3 \
        opencv-python-headless>=4.10 \
        Pillow>=10.4 \
        numpy>=1.26 \
        openai>=1.0.0 \
        python-dotenv>=1.0.0 \
        certifi>=2024.0.0 \
        sahi


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.10-slim

LABEL maintainer="Nikolas de Hor <nikolasdehor79@gmail.com>"
LABEL description="GerminaVision - Analise de germinacao via YOLO11 (CPU)"

# Dependências de runtime do OpenCV headless e da lib GL (sem CUDA, sem xorg)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copia pacotes Python do builder
COPY --from=builder /install /usr/local/lib/python3.10/site-packages

WORKDIR /app

# Copia código-fonte do projeto
COPY . .

# Cria pastas de dados que o Flask espera (volumes serão montados sobre elas)
RUN mkdir -p static/uploads static/results models data

# Força ultralytics a usar apenas CPU e a nao tentar baixar CUDA
ENV ULTRALYTICS_DEVICE=cpu
# Evita download automatico de pesos na inicializacao (o modelo esta em models/)
ENV YOLO_AUTOINSTALL=0
# Flask nao deve rodar em debug em producao
ENV FLASK_ENV=production
# Garante que logs apareçam sem buffer no stdout do container
ENV PYTHONUNBUFFERED=1

EXPOSE 5001

# Usa gunicorn para producao; se quiser dev rapido, troque por: python run.py
CMD ["python", "run.py"]
