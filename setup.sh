#!/usr/bin/env bash
#
# setup.sh - Instalador do GerminaVision em uma máquina nova (Linux ou macOS).
#
# Uso básico:
#   git clone https://github.com/nikolasdehor/projetogerminacao.git
#   cd projetogerminacao
#   bash setup.sh
#
# Rode "bash setup.sh --help" para ver todas as opções.
#
# O script é idempotente: rodar duas vezes não quebra nem duplica nada.
# Compatível com bash 3.2 (macOS) e bash moderno (Linux).

set -euo pipefail

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_NAME="GerminaVision"
APP_PORT="5001"
VENV_DIR="$SCRIPT_DIR/.venv"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"
ENV_FILE="$SCRIPT_DIR/.env"
MODEL_FILE="$SCRIPT_DIR/models/best.pt"
SMOKE_TEST="$SCRIPT_DIR/scripts/ci_smoke.py"
RUNTIME_DIRS="static/uploads static/results models data logs"
PYTHON_MIN_MAJOR=3
PYTHON_MIN_MINOR=10
MODEL_MIN_BYTES=1000000
SYSTEMD_UNIT_FILE="$HOME/.config/systemd/user/germinavision.service"
LAUNCHD_LABEL="com.germinavision.app"
LAUNCHD_PLIST="$HOME/Library/LaunchAgents/$LAUNCHD_LABEL.plist"
APT_PACKAGES="git curl python3 python3-venv python3-pip python3-dev build-essential libgl1 libglib2.0-0"

# ---------------------------------------------------------------------------
# Estado global (flags e resumo)
# ---------------------------------------------------------------------------
ASSUME_YES=0
WITH_SERVICE=0
WITH_TUNNEL=0
START_APP=0
OS_NAME="$(uname -s)"
PYTHON_BIN=""
SUMMARY=""
NEXT_STEPS=""

# ---------------------------------------------------------------------------
# Helpers de saída
# ---------------------------------------------------------------------------
log() {
    printf '[setup] %s\n' "$1"
}

warn() {
    printf '[setup] AVISO: %s\n' "$1" >&2
}

die() {
    printf '[setup] ERRO: %s\n' "$1" >&2
    exit 1
}

add_summary() {
    SUMMARY="${SUMMARY}  - $1\n"
}

add_step() {
    NEXT_STEPS="${NEXT_STEPS}  - $1\n"
}

confirm() {
    # Pergunta sim/não. Com --yes responde sim automaticamente.
    if [ "$ASSUME_YES" = "1" ]; then
        return 0
    fi
    if [ ! -t 0 ]; then
        warn "Entrada não interativa detectada. Use --yes para aceitar automaticamente."
        return 1
    fi
    printf '[setup] %s [s/N] ' "$1"
    read -r resposta
    case "$resposta" in
        s|S|sim|Sim|SIM|y|Y) return 0 ;;
        *) return 1 ;;
    esac
}

usage() {
    cat <<EOF
$APP_NAME - instalador para máquina nova (Linux ou macOS)

Uso:
  bash setup.sh [opções]

Opções:
  --yes            Modo não interativo: assume "sim" em todas as confirmações.
  --with-service   Instala serviço de inicialização automática:
                   systemd user unit no Linux, launchd plist no macOS.
  --with-tunnel    Instala o cloudflared e imprime os passos para criar o
                   tunnel (cloudflared tunnel login etc). Não embute credenciais.
  --start          Sobe a aplicação Flask ao final do setup (porta $APP_PORT).
  --help           Mostra esta ajuda e sai.

O que o script faz:
  1. Verifica pré-requisitos (git, python3 >= $PYTHON_MIN_MAJOR.$PYTHON_MIN_MINOR, pip).
  2. No Linux com apt-get, instala dependências de sistema (pede confirmação).
  3. Cria o ambiente virtual em .venv e instala requirements.txt.
  4. Cria os diretórios de runtime (static/uploads, static/results, models, data, logs).
  5. Gera um .env com placeholders se ele não existir (nunca sobrescreve).
  6. Verifica os pesos do modelo em models/best.pt e orienta a cópia se faltarem.
  7. Roda um smoke test leve (scripts/ci_smoke.py) sem falhar o setup.
EOF
}

# ---------------------------------------------------------------------------
# Parsing de argumentos
# ---------------------------------------------------------------------------
parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --yes)          ASSUME_YES=1 ;;
            --with-service) WITH_SERVICE=1 ;;
            --with-tunnel)  WITH_TUNNEL=1 ;;
            --start)        START_APP=1 ;;
            --help|-h)      usage; exit 0 ;;
            *)              usage >&2; die "Opção desconhecida: $1" ;;
        esac
        shift
    done
}

# ---------------------------------------------------------------------------
# Detecção de sistema operacional
# ---------------------------------------------------------------------------
check_os() {
    case "$OS_NAME" in
        Linux)  log "Sistema detectado: Linux" ;;
        Darwin) log "Sistema detectado: macOS" ;;
        *)      die "Sistema não suportado: $OS_NAME (este script cobre Linux e macOS)" ;;
    esac
}

# ---------------------------------------------------------------------------
# Dependências de sistema (Linux com apt-get)
# ---------------------------------------------------------------------------
run_as_root() {
    if [ "$(id -u)" = "0" ]; then
        "$@"
        return
    fi
    if ! command -v sudo >/dev/null 2>&1; then
        warn "sudo não encontrado. Rode como root ou instale manualmente: $APT_PACKAGES"
        return 1
    fi
    sudo "$@"
}

install_system_deps() {
    if [ "$OS_NAME" != "Linux" ]; then
        return 0
    fi
    if ! command -v apt-get >/dev/null 2>&1; then
        warn "apt-get não disponível nesta distribuição."
        warn "Instale manualmente os equivalentes de: $APT_PACKAGES"
        return 0
    fi
    log "Pacotes de sistema necessários: $APT_PACKAGES"
    if ! confirm "Instalar pacotes de sistema via apt-get (usa sudo)?"; then
        warn "Pulando pacotes de sistema. Se o pip falhar depois, instale-os e rode o setup de novo."
        return 0
    fi
    if run_as_root apt-get update && run_as_root apt-get install -y $APT_PACKAGES; then
        add_summary "Pacotes de sistema instalados via apt-get"
        return 0
    fi
    warn "Falha ao instalar pacotes de sistema. Verifique sua conexão e permissões."
    return 0
}

# ---------------------------------------------------------------------------
# Pré-requisitos: git, python3 e pip
# ---------------------------------------------------------------------------
check_git() {
    if command -v git >/dev/null 2>&1; then
        return 0
    fi
    if [ "$OS_NAME" = "Darwin" ]; then
        die "git não encontrado. Instale com: xcode-select --install (ou brew install git)"
    fi
    die "git não encontrado. Instale com: sudo apt-get install -y git"
}

check_python() {
    if ! command -v python3 >/dev/null 2>&1; then
        if [ "$OS_NAME" = "Darwin" ]; then
            die "python3 não encontrado. Instale com: brew install python@3.12"
        fi
        die "python3 não encontrado. Instale com: sudo apt-get install -y python3 python3-venv python3-pip"
    fi
    PYTHON_BIN="$(command -v python3)"
    if ! "$PYTHON_BIN" -c "import sys; raise SystemExit(0 if sys.version_info >= ($PYTHON_MIN_MAJOR, $PYTHON_MIN_MINOR) else 1)"; then
        versao="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
        die "Python $versao encontrado, mas o projeto exige $PYTHON_MIN_MAJOR.$PYTHON_MIN_MINOR ou superior. Atualize o Python e rode o setup de novo."
    fi
    log "Python OK: $("$PYTHON_BIN" --version 2>&1) em $PYTHON_BIN"
}

check_pip() {
    if "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
        return 0
    fi
    if "$PYTHON_BIN" -m ensurepip --version >/dev/null 2>&1; then
        return 0
    fi
    if [ "$OS_NAME" = "Darwin" ]; then
        die "pip não disponível no python3. Reinstale o Python via Homebrew: brew reinstall python@3.12"
    fi
    die "pip não disponível no python3. Instale com: sudo apt-get install -y python3-pip python3-venv"
}

# ---------------------------------------------------------------------------
# Ambiente virtual e dependências Python
# ---------------------------------------------------------------------------
create_venv() {
    if [ -x "$VENV_DIR/bin/python" ]; then
        log "Ambiente virtual já existe em .venv, mantendo."
        return 0
    fi
    log "Criando ambiente virtual em .venv ..."
    if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
        die "Falha ao criar o venv. No Linux, instale: sudo apt-get install -y python3-venv"
    fi
    add_summary "Ambiente virtual criado em .venv"
}

install_requirements() {
    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        die "requirements.txt não encontrado em $SCRIPT_DIR. Confira se o clone do repositório está completo."
    fi
    log "Atualizando pip e instalando requirements.txt (torch e ultralytics são pesados, pode demorar) ..."
    "$VENV_DIR/bin/pip" install --upgrade pip --quiet
    "$VENV_DIR/bin/pip" install -r "$REQUIREMENTS_FILE"
    add_summary "Dependências Python instaladas a partir de requirements.txt"
}

# ---------------------------------------------------------------------------
# Diretórios de runtime
# ---------------------------------------------------------------------------
create_runtime_dirs() {
    for dir in $RUNTIME_DIRS; do
        mkdir -p "$SCRIPT_DIR/$dir"
    done
    log "Diretórios de runtime garantidos: $RUNTIME_DIRS"
    add_summary "Diretórios de runtime criados ou já existentes"
}

# ---------------------------------------------------------------------------
# Arquivo .env
# ---------------------------------------------------------------------------
create_env_file() {
    if [ -f "$ENV_FILE" ]; then
        log ".env já existe, não será sobrescrito."
        return 0
    fi
    log "Gerando .env com placeholders (preencha antes de usar o WhatsApp) ..."
    cat > "$ENV_FILE" <<'ENVEOF'
# =============================================================================
# GerminaVision - variáveis de ambiente
# Preencha os valores marcados como OBRIGATÓRIA antes de usar o recurso.
# Este arquivo não é versionado (.env* está no .gitignore).
# =============================================================================

# --- Chat agrícola com LLM (opcional) ---------------------------------------
# Chave da OpenRouter. Sem ela o chat com IA fica inativo, o resto funciona.
OPENROUTER_API_KEY=

# --- Webhook do WhatsApp (OBRIGATÓRIAS para o bot funcionar) -----------------
# URL pública onde este app fica acessível (cloudflared tunnel ou domínio).
# OBRIGATÓRIA para WhatsApp. Exemplo: https://wpp.seudominio.com.br
PUBLIC_BASE_URL=

# Endpoint da Evolution API (VPS ou SaaS onde ela roda).
# OBRIGATÓRIA para WhatsApp. Exemplo: https://sua-instancia.evolution-api.com
EVOLUTION_API_URL=

# Chave de autenticação gerada no painel da Evolution API.
# OBRIGATÓRIA para WhatsApp.
EVOLUTION_API_KEY=

# Nome da instância configurada na Evolution API.
# OBRIGATÓRIA para WhatsApp (padrão usado no projeto: germinavision).
EVOLUTION_INSTANCE_NAME=germinavision

# --- WhatsApp (opcionais) ----------------------------------------------------
# Host header customizado quando a rede bloqueia DNS dinâmico (proxy por IP).
EVOLUTION_API_HOST_HEADER=

# Verificação SSL ao falar com a Evolution API (true ou false, padrão true).
# Use false apenas em ambiente local atrás de proxy ou CA privada.
EVOLUTION_SSL_VERIFY=true

# Número do bot sem + e sem espaços (detecta @mention em grupos).
EVOLUTION_INSTANCE_PHONE=

# Modo de resposta em grupos: off, mention_only, image_always_text_mention, all.
GROUP_RESPONSE_MODE=image_always_text_mention

# Whitelist de grupos (CSV de JIDs completos com @g.us). Vazio = todos.
ALLOWED_GROUPS=

# --- Análise de germinação ---------------------------------------------------
# Capacidade padrão da bandeja (número de células).
TRAY_CAPACITY=200

# Guard de detecção: mínimo de detecções para aceitar uma análise.
GUARD_MIN_DETECTIONS=3

# Guard de detecção: confiança média mínima para aceitar uma análise.
GUARD_MIN_MEAN_CONF=0.55

# --- Flags de inferência (avançadas, padrão 0 = desligado) --------------------
# Força normalização de iluminação antes da inferência.
GERMINAVISION_FORCE_NORMALIZE=0

# Desativa a reconstrução de cor em fotos com LED magenta (usa imagem crua).
GERMINAVISION_RAW_MAGENTA=0

# Usa a versão legada da reconstrução de cor.
GERMINAVISION_LEGACY_RECONSTRUCT=0

# Split de detecções muito grandes (megabox). Padrão ligado, 0 desativa.
GERMINAVISION_SPLIT_MEGABOX=1

# Gate de segurança para contagem híbrida de células.
GERMINAVISION_HYBRID_CELL_SAFETY=0

# Fallback por componente verde em fotos com LED roxo ou magenta.
GERMINAVISION_GREEN_COMPONENT_FALLBACK=0

# Fallback híbrido em cenas magenta.
GERMINAVISION_HYBRID_FALLBACK=0

# Naturaliza a imagem de saída em cenas magenta.
GERMINAVISION_NATURALIZE_MAGENTA=0

# --- SSL (opcional) -----------------------------------------------------------
# Caminho de um bundle de certificados customizado. Vazio = usa o do certifi.
SSL_CERT_FILE=
ENVEOF
    add_summary ".env gerado com placeholders"
    add_step "Preencher o .env com as credenciais reais (PUBLIC_BASE_URL, EVOLUTION_API_URL, EVOLUTION_API_KEY, EVOLUTION_INSTANCE_NAME e, se quiser chat com IA, OPENROUTER_API_KEY)"
}

# ---------------------------------------------------------------------------
# Pesos do modelo YOLO
# ---------------------------------------------------------------------------
file_size_bytes() {
    wc -c < "$1" | tr -d '[:space:]'
}

download_model_weights() {
    # Não existe URL pública conhecida para os pesos treinados. Se você hospedar
    # o best.pt em algum lugar, exporte GERMINAVISION_WEIGHTS_URL antes de rodar.
    url="${GERMINAVISION_WEIGHTS_URL:-}"
    if [ -z "$url" ]; then
        return 1
    fi
    if ! command -v curl >/dev/null 2>&1; then
        warn "curl não encontrado, não foi possível baixar os pesos de GERMINAVISION_WEIGHTS_URL."
        return 1
    fi
    log "Baixando pesos do modelo de GERMINAVISION_WEIGHTS_URL ..."
    tmp_file="$(mktemp)"
    if ! curl -fL --retry 3 -o "$tmp_file" "$url"; then
        rm -f "$tmp_file"
        warn "Download dos pesos falhou. Verifique a URL em GERMINAVISION_WEIGHTS_URL."
        return 1
    fi
    if [ "$(file_size_bytes "$tmp_file")" -lt "$MODEL_MIN_BYTES" ]; then
        rm -f "$tmp_file"
        warn "Arquivo baixado é pequeno demais para ser um peso YOLO válido. Download descartado."
        return 1
    fi
    mv "$tmp_file" "$MODEL_FILE"
    add_summary "Pesos do modelo baixados para models/best.pt"
    return 0
}

check_model_weights() {
    if [ -f "$MODEL_FILE" ] && [ "$(file_size_bytes "$MODEL_FILE")" -ge "$MODEL_MIN_BYTES" ]; then
        log "Pesos do modelo encontrados em models/best.pt"
        add_summary "Pesos do modelo presentes em models/best.pt"
        return 0
    fi
    if download_model_weights; then
        return 0
    fi
    warn "models/best.pt não encontrado (o arquivo é ignorado pelo git e não vem no clone)."
    cat <<EOF
[setup] Como obter os pesos treinados:
[setup]   1. Copie do Mac atual via scp (ajuste usuário e IP):
[setup]      scp nikolas@IP_DO_MAC:"/Users/nikolas/projetogerminação/models/best.pt" "$MODEL_FILE"
[setup]   2. Ou baixe do Google Drive usado no retreino do Colab
[setup]      (MyDrive/projetogerminacao/runs/train/) e salve em models/best.pt
[setup]   3. Ou exporte GERMINAVISION_WEIGHTS_URL com uma URL sua e rode o setup de novo.
[setup] Sem o best.pt o app cai no fallback yolo11s.pt (COCO genérico), com
[setup] precisão bem menor para mudas.
EOF
    add_step "Copiar os pesos treinados para models/best.pt (sem eles o app usa o fallback COCO, menos preciso)"
    return 0
}

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
run_smoke_test() {
    if [ ! -f "$SMOKE_TEST" ]; then
        log "Smoke test não encontrado (scripts/ci_smoke.py), pulando."
        return 0
    fi
    log "Rodando smoke test leve (scripts/ci_smoke.py) ..."
    if "$VENV_DIR/bin/python" "$SMOKE_TEST"; then
        log "Smoke test passou."
        add_summary "Smoke test executado com sucesso"
        return 0
    fi
    warn "Smoke test falhou, mas o setup continua. Isso pode acontecer se faltarem credenciais ou pesos do modelo."
    add_step "Investigar a falha do smoke test: $VENV_DIR/bin/python scripts/ci_smoke.py"
    return 0
}

# ---------------------------------------------------------------------------
# Serviço de inicialização (--with-service)
# ---------------------------------------------------------------------------
install_service_linux() {
    if [ -f "$SYSTEMD_UNIT_FILE" ]; then
        log "Unit systemd já existe em $SYSTEMD_UNIT_FILE, não será sobrescrita."
    else
        mkdir -p "$(dirname "$SYSTEMD_UNIT_FILE")"
        cat > "$SYSTEMD_UNIT_FILE" <<EOF
[Unit]
Description=$APP_NAME - app Flask de monitoramento de germinação
After=network-online.target

[Service]
WorkingDirectory=$SCRIPT_DIR
ExecStart=$VENV_DIR/bin/python -m flask --app run:app run --host 0.0.0.0 --port $APP_PORT
Environment=PYTHONUNBUFFERED=1
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
EOF
        log "Unit systemd criada em $SYSTEMD_UNIT_FILE"
        add_summary "Serviço systemd (user unit) criado: germinavision.service"
    fi
    cat <<EOF
[setup] Para ativar o serviço no Linux:
[setup]   systemctl --user daemon-reload
[setup]   systemctl --user enable --now germinavision.service
[setup]   loginctl enable-linger $USER   # mantém o serviço rodando sem sessão aberta
[setup] Logs: journalctl --user -u germinavision.service -f
EOF
    add_step "Ativar o serviço: systemctl --user enable --now germinavision.service (e loginctl enable-linger)"
}

install_service_macos() {
    if [ -f "$LAUNCHD_PLIST" ]; then
        log "Plist launchd já existe em $LAUNCHD_PLIST, não será sobrescrito."
    else
        mkdir -p "$(dirname "$LAUNCHD_PLIST")"
        cat > "$LAUNCHD_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LAUNCHD_LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$VENV_DIR/bin/python</string>
        <string>-m</string>
        <string>flask</string>
        <string>--app</string>
        <string>run:app</string>
        <string>run</string>
        <string>--host</string>
        <string>0.0.0.0</string>
        <string>--port</string>
        <string>$APP_PORT</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$SCRIPT_DIR</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$SCRIPT_DIR/logs/germinavision-app.out.log</string>
    <key>StandardErrorPath</key>
    <string>$SCRIPT_DIR/logs/germinavision-app.err.log</string>
</dict>
</plist>
EOF
        log "Plist launchd criado em $LAUNCHD_PLIST"
        add_summary "Serviço launchd criado: $LAUNCHD_LABEL"
    fi
    cat <<EOF
[setup] Para ativar o serviço no macOS:
[setup]   launchctl load -w "$LAUNCHD_PLIST"
[setup] Para parar:
[setup]   launchctl unload "$LAUNCHD_PLIST"
[setup] Logs em: $SCRIPT_DIR/logs/germinavision-app.out.log e .err.log
EOF
    add_step "Ativar o serviço: launchctl load -w $LAUNCHD_PLIST"
}

install_service() {
    if [ "$WITH_SERVICE" != "1" ]; then
        return 0
    fi
    if [ "$OS_NAME" = "Linux" ]; then
        install_service_linux
        return 0
    fi
    install_service_macos
}

# ---------------------------------------------------------------------------
# Cloudflare tunnel (--with-tunnel)
# ---------------------------------------------------------------------------
install_cloudflared_linux() {
    arch="$(uname -m)"
    case "$arch" in
        x86_64|amd64)  cf_arch="amd64" ;;
        aarch64|arm64) cf_arch="arm64" ;;
        *)
            warn "Arquitetura $arch sem binário pronto. Veja: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
            return 0
            ;;
    esac
    if ! command -v curl >/dev/null 2>&1; then
        warn "curl não encontrado, não foi possível baixar o cloudflared."
        return 0
    fi
    url="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$cf_arch"
    tmp_bin="$(mktemp)"
    log "Baixando cloudflared ($cf_arch) ..."
    if ! curl -fL --retry 3 -o "$tmp_bin" "$url"; then
        rm -f "$tmp_bin"
        warn "Download do cloudflared falhou. Instale manualmente e rode o setup de novo."
        return 0
    fi
    chmod +x "$tmp_bin"
    if confirm "Instalar cloudflared em /usr/local/bin (usa sudo)?"; then
        if run_as_root mv "$tmp_bin" /usr/local/bin/cloudflared; then
            add_summary "cloudflared instalado em /usr/local/bin"
            return 0
        fi
    fi
    mkdir -p "$HOME/.local/bin"
    mv "$tmp_bin" "$HOME/.local/bin/cloudflared"
    warn "cloudflared instalado em ~/.local/bin. Garanta que ~/.local/bin está no seu PATH."
    add_summary "cloudflared instalado em ~/.local/bin"
}

install_cloudflared_macos() {
    if ! command -v brew >/dev/null 2>&1; then
        warn "Homebrew não encontrado. Instale o cloudflared manualmente: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
        return 0
    fi
    log "Instalando cloudflared via Homebrew ..."
    if brew install cloudflared; then
        add_summary "cloudflared instalado via Homebrew"
    else
        warn "brew install cloudflared falhou. Instale manualmente e rode o setup de novo."
    fi
}

setup_tunnel() {
    if [ "$WITH_TUNNEL" != "1" ]; then
        return 0
    fi
    if command -v cloudflared >/dev/null 2>&1; then
        log "cloudflared já instalado: $(cloudflared --version 2>/dev/null | head -n 1)"
    elif [ "$OS_NAME" = "Linux" ]; then
        install_cloudflared_linux
    else
        install_cloudflared_macos
    fi
    cat <<EOF
[setup] Passos para criar o tunnel da Cloudflare (sem credenciais embutidas):
[setup]   1. cloudflared tunnel login
[setup]      (abre o navegador, escolha a zona do seu domínio)
[setup]   2. cloudflared tunnel create germinavision
[setup]      (gera o JSON de credenciais em ~/.cloudflared/)
[setup]   3. Crie ~/.cloudflared/config.yml com:
[setup]        tunnel: ID_DO_TUNNEL
[setup]        credentials-file: CAMINHO_DO_JSON_GERADO
[setup]        ingress:
[setup]          - hostname: wpp.seudominio.com.br
[setup]            service: http://localhost:$APP_PORT
[setup]          - service: http_status:404
[setup]   4. cloudflared tunnel route dns germinavision wpp.seudominio.com.br
[setup]   5. cloudflared tunnel run germinavision
[setup]   6. Atualize PUBLIC_BASE_URL no .env com https://wpp.seudominio.com.br
[setup] Alternativa rápida sem domínio (URL temporária que muda a cada execução):
[setup]   cloudflared tunnel --url http://localhost:$APP_PORT
EOF
    add_step "Configurar o tunnel da Cloudflare (cloudflared tunnel login, create, route dns) e atualizar PUBLIC_BASE_URL no .env"
}

# ---------------------------------------------------------------------------
# Resumo final e start
# ---------------------------------------------------------------------------
print_summary() {
    printf '\n[setup] ============================================================\n'
    printf '[setup] %s - setup concluído\n' "$APP_NAME"
    printf '[setup] ============================================================\n'
    if [ -n "$SUMMARY" ]; then
        printf '[setup] O que foi feito:\n'
        printf '%b' "$SUMMARY"
    fi
    if [ -n "$NEXT_STEPS" ]; then
        printf '[setup] Próximos passos manuais:\n'
        printf '%b' "$NEXT_STEPS"
    fi
    printf '[setup] Para subir a aplicação manualmente:\n'
    printf '  %s/bin/python -m flask --app run:app run --host 0.0.0.0 --port %s\n' "$VENV_DIR" "$APP_PORT"
    printf '[setup] Depois acesse http://localhost:%s e, para conectar o WhatsApp,\n' "$APP_PORT"
    printf '[setup] http://localhost:%s/whatsapp (escanear o QR Code da Evolution API).\n' "$APP_PORT"
}

start_app() {
    if [ "$START_APP" != "1" ]; then
        return 0
    fi
    log "Subindo a aplicação na porta $APP_PORT (Ctrl+C para parar) ..."
    cd "$SCRIPT_DIR"
    exec "$VENV_DIR/bin/python" -m flask --app run:app run --host 0.0.0.0 --port "$APP_PORT"
}

# ---------------------------------------------------------------------------
# Fluxo principal
# ---------------------------------------------------------------------------
main() {
    parse_args "$@"
    log "Iniciando setup do $APP_NAME em $SCRIPT_DIR"
    check_os
    install_system_deps
    check_git
    check_python
    check_pip
    create_venv
    install_requirements
    create_runtime_dirs
    create_env_file
    check_model_weights
    run_smoke_test
    install_service
    setup_tunnel
    print_summary
    start_app
}

main "$@"
