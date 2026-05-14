#!/bin/bash
# GerminaVision watchdog - monitora Flask + cloudflared e reinicia se cair
set -u

LOG=/tmp/germinavision_watchdog.log
PROJECT_DIR=/Users/nikolas/projetogerminação
CLOUDFLARED_LOG=/tmp/cloudflared.log
FLASK_LOG=/tmp/flask.log
EVOLUTION_URL="https://your-evolution-instance.example.com"
EVOLUTION_KEY="REDACTED_KEY"
INSTANCE="germinavision"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

start_flask() {
    log "Iniciando Flask..."
    cd "$PROJECT_DIR" && source venv/bin/activate && nohup python run.py > "$FLASK_LOG" 2>&1 &
    sleep 6
}

start_cloudflared() {
    log "Iniciando cloudflared..."
    nohup cloudflared tunnel --url http://localhost:5001 > "$CLOUDFLARED_LOG" 2>&1 &
    sleep 8
    # Pega URL nova
    URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$CLOUDFLARED_LOG" | tail -1)
    if [ -n "$URL" ]; then
        log "Nova URL pública: $URL"
        register_webhook "$URL"
    else
        log "ERRO: não conseguiu extrair URL do cloudflared"
    fi
}

register_webhook() {
    local URL="$1"
    log "Registrando webhook em $URL/api/whatsapp/webhook"
    curl -s -X POST "$EVOLUTION_URL/webhook/set/$INSTANCE" \
      -H "Content-Type: application/json" \
      -H "apikey: $EVOLUTION_KEY" \
      -d "{\"webhook\":{\"enabled\":true,\"url\":\"$URL/api/whatsapp/webhook\",\"webhookByEvents\":false,\"webhookBase64\":true,\"events\":[\"MESSAGES_UPSERT\",\"CONNECTION_UPDATE\"]}}" \
      > /dev/null
    log "Webhook atualizado"
}

check_flask() {
    curl -sf --max-time 5 http://localhost:5001/api/whatsapp/status > /dev/null
}

check_cloudflared() {
    pgrep -f "cloudflared tunnel" > /dev/null
}

check_disk() {
    avail_mb=$(df -m / | awk 'NR==2{print $4}')
    if [ "$avail_mb" -lt 4096 ]; then
        log "DISCO BAIXO: ${avail_mb} MB. Limpando caches..."
        rm -rf ~/.cache/uv ~/Library/Caches/Google ~/.cache/codex-runtimes 2>/dev/null
        rm -rf ~/Library/Caches/camoufox ~/Library/Caches/ms-playwright ~/Library/Caches/ms-playwright-go 2>/dev/null
        rm -rf ~/.cache/whisper ~/Library/Caches/Homebrew 2>/dev/null
        brew cleanup --prune=all 2>/dev/null
        new_avail=$(df -m / | awk 'NR==2{print $4}')
        log "Após limpeza: ${new_avail} MB livres"
    fi
}

check_training() {
    TRAIN_PID=$(pgrep -f "python train.py" | head -1)
    DONE_FLAG=/tmp/train_done.flag

    if [ -z "$TRAIN_PID" ] && [ ! -f "$DONE_FLAG" ]; then
        if grep -qE "Results saved to|Training complete|stopping training" /tmp/train.log 2>/dev/null; then
            BEST_MAP50=$(awk -F',' 'NR>1 {if($7>max) max=$7} END {print max}' /Users/nikolas/projetogerminação/runs/morango_v2/results.csv 2>/dev/null)
            LAST_EPOCH=$(awk -F',' 'NR>1 {n=$1} END {print n}' /Users/nikolas/projetogerminação/runs/morango_v2/results.csv 2>/dev/null)
            log "TREINO TERMINOU. Epochs: $LAST_EPOCH, melhor mAP50: $BEST_MAP50"
            echo "OK|epoch=$LAST_EPOCH|mAP50=$BEST_MAP50|$(date)" > "$DONE_FLAG"
            osascript -e "display notification \"Treino terminou! mAP50: $BEST_MAP50 em $LAST_EPOCH epochs. Veja models/best.pt\" with title \"GerminaVision\" sound name \"Glass\"" 2>/dev/null
            if [ -f /Users/nikolas/projetogerminação/runs/morango_v2/weights/best.pt ]; then
                cp /Users/nikolas/projetogerminação/runs/morango_v2/weights/best.pt /Users/nikolas/projetogerminação/models/best.pt
                log "best.pt copiado para models/"
                osascript -e "display notification \"models/best.pt atualizado, Flask vai recarregar\" with title \"GerminaVision\"" 2>/dev/null
            fi
        elif ! pgrep -f "python train.py" > /dev/null; then
            log "TREINO CRASHOU sem finalizar. Relançando..."
            cd /Users/nikolas/projetogerminação && nohup caffeinate -i venv/bin/python train.py > /tmp/train.log 2>&1 &
            sleep 3
            NEW_PID=$(pgrep -f "python train.py" | head -1)
            log "Relançado, novo PID: $NEW_PID"
            osascript -e "display notification \"Treino crashou e foi relançado (PID $NEW_PID). Continuando...\" with title \"GerminaVision\"" 2>/dev/null
        fi
    fi
}

log "==== Watchdog iniciado ===="

while true; do
    if ! check_flask; then
        log "Flask CAIU — reiniciando"
        pkill -f "python run.py" 2>/dev/null
        sleep 2
        start_flask
    fi
    if ! check_cloudflared; then
        log "cloudflared CAIU — reiniciando"
        start_cloudflared
    fi
    check_disk
    check_training
    sleep 60
done
