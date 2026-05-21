'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
let selectedFile  = null;
let temporalChart = null;
const CLASS_COLORS   = { Germinacao:'#2f6f4e', Folha:'#b77433' };
const CLASS_DISPLAY  = { Germinacao: 'Germinação', Folha: 'Folha' };
const displayClass   = (c) => CLASS_DISPLAY[c] || c;

// ── DOM ───────────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const dropZone     = $('dropZone');
const fileInput    = $('fileInput');
const previewWrap  = $('previewWrap');
const previewImg   = $('previewImg');
const analyzeBtn   = $('analyzeBtn');
const progressWrap = $('progressWrap');
const progressFill = $('progressFill');
const progressLbl  = $('progressLabel');
const confSlider   = $('confSlider');
const confValue    = $('confValue');

// ── Utils ─────────────────────────────────────────────────────────────────────
function showToast(msg, type='info', ms=3200) {
  const t = $('toast');
  t.textContent = msg;
  t.className = `toast show ${type}`;
  clearTimeout(t._t);
  t._t = setTimeout(() => { t.className = 'toast'; }, ms);
}
function setProgress(pct, label) {
  progressFill.style.width = pct + '%';
  progressLbl.textContent  = label;
}
function now() {
  return new Date().toLocaleTimeString('pt-BR', { hour:'2-digit', minute:'2-digit' });
}
function formatTs(ts) {
  // "2026-05-06 22:05:35" → "06/05 22:05"
  const [date, time] = (ts || '').split(' ');
  if (!date || !time) return ts;
  const [y, m, d] = date.split('-');
  const [h, min] = time.split(':');
  return `${d}/${m} ${h}:${min}`;
}
function formatPhone(num) {
  if (!num) return '';
  const s = String(num).replace(/\D/g, '');
  if (s.length === 13) return `+${s.slice(0,2)} ${s.slice(2,4)} ${s.slice(4,9)}-${s.slice(9)}`;
  if (s.length === 12) return `+${s.slice(0,2)} ${s.slice(2,4)} ${s.slice(4,8)}-${s.slice(8)}`;
  if (s.length === 11) return `(${s.slice(0,2)}) ${s.slice(2,7)}-${s.slice(7)}`;
  return num;
}

// ── Status ────────────────────────────────────────────────────────────────────
async function checkStatus() {
  try {
    const d = await fetch('/api/status').then(r => r.json());
    const dot  = $('statusDot'), txt = $('statusText');
    if (d.model_loaded) {
      dot.className = 'status-dot ok';
      txt.textContent = d.custom_model ? 'Modelo personalizado' : 'Modelo COCO';
    } else {
      dot.className = 'status-dot error';
      txt.textContent = 'Modelo offline';
    }
  } catch { $('statusDot').className = 'status-dot error'; $('statusText').textContent = 'Servidor offline'; }
}

// ── File handling ─────────────────────────────────────────────────────────────
function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) { showToast('Selecione uma imagem válida.', 'error'); return; }
  selectedFile = file;
  previewImg.src = URL.createObjectURL(file);
  previewWrap.hidden = false;
  dropZone.hidden    = true;
  analyzeBtn.disabled = false;
}
function clearFile() {
  selectedFile = null; previewImg.src = '';
  previewWrap.hidden = true; dropZone.hidden = false;
  analyzeBtn.disabled = true; fileInput.value = '';
}
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', e => { if (e.key==='Enter'||e.key===' ') fileInput.click(); });
fileInput.addEventListener('change', e => { if (e.target.files[0]) handleFile(e.target.files[0]); });
$('clearImg').addEventListener('click', clearFile);
dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', ()  => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => { e.preventDefault(); dropZone.classList.remove('drag-over'); if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); });
confSlider.addEventListener('input', () => { confValue.textContent = confSlider.value + '%'; });

// ── Analyze ───────────────────────────────────────────────────────────────────
analyzeBtn.addEventListener('click', runAnalysis);

async function runAnalysis() {
  if (!selectedFile) return;
  analyzeBtn.disabled = true;
  analyzeBtn.classList.add('loading');
  analyzeBtn.querySelector('.btn-icon').textContent = '...';
  progressWrap.hidden = false;
  setProgress(15, 'Enviando imagem...');

  const fd = new FormData();
  fd.append('image', selectedFile);
  fd.append('conf',  confSlider.value / 100);
  try {
    setProgress(45, 'Rodando YOLO11...');
    const res  = await fetch('/api/analyze', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Erro desconhecido');
    setProgress(100, 'Concluído!');
    setTimeout(() => { progressWrap.hidden = true; setProgress(0,''); }, 700);

    renderResult(data);
    await loadHistory();
    await loadTemporal();
    showToast(`${data.total_detected} detecção(ões), ${data.germination_rate}% germinação`, 'success');
  } catch(err) {
    setProgress(0,''); progressWrap.hidden = true;
    showToast(err.message, 'error', 5000);
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.classList.remove('loading');
    analyzeBtn.querySelector('.btn-icon').textContent = '>';
  }
}

// ── Render result ─────────────────────────────────────────────────────────────
function renderResult(data) {
  // Metrics side
  $('metricsSide').hidden = false;
  $('mRate').textContent      = data.germination_rate + '%';
  $('mRateBar').style.width   = data.germination_rate + '%';
  const mRateSub = $('mRateSub');
  if (mRateSub) {
    const cap = data.cells_detected || data.tray_capacity || 200;
    mRateSub.textContent = `${data.germinated} de ${cap} células`;
  }
  $('mTotal').textContent     = data.total_detected;
  $('mGerminated').textContent= data.germinated;
  $('mLeaf').textContent      = data.leaf_avg;
  $('inferenceTime').textContent = `Tempo: ${data.inference_time_s}s`;

  // Capacidade detectada automaticamente
  const cellsEl = $('mCells');
  if (cellsEl) {
    const cap = data.cells_detected || data.tray_capacity || 200;
    const isAuto = data.cells_detected && data.cells_detected !== (data.tray_capacity || 200);
    cellsEl.textContent = cap + (isAuto ? ' (auto)' : '');
  }

  // Detection list
  const list = $('detectionList');
  list.innerHTML = '';
  data.detections.forEach(d => {
    const color = CLASS_COLORS[d.class] || '#7b8879';
    const item = document.createElement('div');
    item.className = 'det-item';
    const idLabel = d.plant_id ? `#${d.plant_id} · ` : '';
    item.innerHTML = `
      <span class="det-dot" style="background:${color}"></span>
      <span class="det-name ${d.germinated?'det-ok':'det-no'}">${idLabel}${displayClass(d.class)}</span>
      <span class="det-leaves">${d.leaf_count} folha${d.leaf_count!==1?'s':''}</span>
      <span class="det-conf">${(d.confidence*100).toFixed(0)}%</span>`;
    list.appendChild(item);
  });

  // Full-width result image
  const section = $('resultSection');
  const img     = $('resultImg');
  const dl      = $('resultDownload');
  img.src   = data.result_image + '?t=' + Date.now();
  img.alt   = `Detecções: ${data.total_detected} mudas`;
  dl.href   = data.result_image;
  section.hidden = false;
  section.classList.add('fade-in');

  // Lightbox on click
  img.parentElement.onclick = () => openLightbox(data.result_image);

  // Scroll suave para o resultado
  setTimeout(() => section.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);

  // Hero stats
  loadHistory(true);
}

// ── Lightbox ──────────────────────────────────────────────────────────────────
function openLightbox(src) {
  const img = $('lightboxImg');
  img.src = '';                          // reset first to force reload
  img.src = src.split('?')[0] + '?t=' + Date.now();
  $('lightbox').hidden = false;
  document.body.style.overflow = 'hidden';
}
function closeLightbox() {
  $('lightbox').hidden = true;
  $('lightboxImg').src = '';
  document.body.style.overflow = '';
}
$('lightboxClose').addEventListener('click', closeLightbox);
$('lightbox').addEventListener('click', e => { if (e.target === $('lightbox')) closeLightbox(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

// ── History ───────────────────────────────────────────────────────────────────
const HISTORY_PAGE = 20;
let _historyOffset = 0;
let _historyTotal  = 0;
let _historyAllRows = [];  // acumula registros já carregados

async function loadHistory(reset = true) {
  if (reset) {
    _historyOffset  = 0;
    _historyAllRows = [];
  }
  try {
    const data = await fetch(`/api/history?limit=${HISTORY_PAGE}&offset=${_historyOffset}`).then(r => r.json());
    _historyTotal   = data.total;
    _historyOffset += data.items.length;
    _historyAllRows = _historyAllRows.concat(data.items);
    renderHistory(_historyAllRows, data.total);
    if (reset) updateHeroStats(data.total, data.items);
  } catch { /* silencioso */ }
}

async function loadMoreHistory() {
  if (_historyOffset >= _historyTotal) return;
  await loadHistory(false);
}

function renderHistory(records, total) {
  $('historyCount').textContent = `${total} registro${total!==1?'s':''}`;
  const body = $('historyBody');
  if (!records.length) {
    body.innerHTML = '<tr class="empty-row"><td colspan="10">Nenhuma análise ainda. Faça o primeiro upload.</td></tr>';
    _updateLoadMoreBtn(0, 0);
    return;
  }
  body.innerHTML = records.map(r => `
    <tr>
      <td style="font-family:var(--font-mono);color:var(--text-3)">${r.id}</td>
      <td>${formatTs(r.timestamp)}</td>
      <td style="max-width:140px;overflow:hidden;text-overflow:ellipsis" title="${r.filename}">${r.filename}</td>
      <td>${r.total_detected}</td>
      <td>${r.germinated}</td>
      <td><span class="pill ${r.germination_rate>=50?'pill-green':'pill-red'}">${r.germination_rate}%</span></td>
      <td>${r.leaf_avg}</td>
      <td><span class="badge-source" title="${r.source==='whatsapp'?'Análise via WhatsApp':'Análise via dashboard'}">${r.source==='whatsapp' ? 'WhatsApp' + (r.sender ? ' ' + formatPhone(r.sender) : '') : 'Web'}</span></td>
      <td>${r.result_image ? `<img src="${r.result_image}" class="result-thumb" alt="resultado" onclick="openLightbox('${r.result_image}')" />` : '-'}</td>
      <td><button class="btn-del" title="Deletar" onclick="deleteRecord(${r.id})">Excluir</button></td>
    </tr>`).join('');
  _updateLoadMoreBtn(_historyOffset, total);
}

function _updateLoadMoreBtn(loaded, total) {
  const btn = $('loadMore');
  if (!btn) return;
  const remaining = total - loaded;
  if (remaining > 0) {
    btn.hidden = false;
    btn.textContent = `Carregar mais (${remaining} restante${remaining!==1?'s':''})`;
  } else {
    btn.hidden = true;
  }
}

function updateHeroStats(total, recentItems) {
  $('statTotal').textContent = total;
  if (!recentItems.length) { $('statGermRate').textContent = '—'; $('statLeafAvg').textContent = '—'; return; }
  $('statGermRate').textContent = (recentItems.reduce((s,r) => s+r.germination_rate,0)/recentItems.length).toFixed(1) + '%';
  $('statLeafAvg').textContent  = (recentItems.reduce((s,r) => s+r.leaf_avg,0)/recentItems.length).toFixed(1);
}

async function deleteRecord(id) {
  if (!confirm('Deletar este registro?')) return;
  await fetch(`/api/history/${id}`, { method: 'DELETE' });
  showToast('Registro deletado.', 'success');
  await loadHistory(true);
  await loadTemporal();
}

// ── Temporal + Stats ──────────────────────────────────────────────────────────
let distChart = null;

async function loadTemporal() {
  try {
    const [temporal, statsData] = await Promise.all([
      fetch('/api/temporal').then(r => r.json()),
      fetch('/api/stats').then(r => r.json()),
    ]);
    renderChart(temporal);
    renderStats(statsData.summary || {});
    renderDistChart(statsData.distribution || []);
  } catch { /* silencioso */ }
}

function renderStats(s) {
  if (!s.total) return;
  $('sumTotal').textContent    = s.total ?? '—';
  $('sumAvg').textContent      = s.avg_rate != null ? s.avg_rate + '%' : '—';
  $('sumBest').textContent     = s.best_rate != null ? s.best_rate + '%' : '—';
  $('sumDetected').textContent = s.total_detected ?? '—';
  $('sumLeaves').textContent   = s.avg_leaves ?? '—';
  $('sumDays').textContent     = s.days_tracked ?? '—';
}

function renderDistChart(distribution) {
  const empty = $('distEmpty');
  const legend = $('distLegend');
  if (!distribution.length) { empty.hidden = false; return; }
  empty.hidden = true;

  const COLORS = { 'Ótima (≥80%)': '#2f6f4e', 'Boa (60-79%)': '#426f92', 'Regular (40-59%)': '#b77433', 'Baixa (<40%)': '#b64f45' };
  const labels = distribution.map(d => d.faixa);
  const values = distribution.map(d => d.qtd);
  const colors = labels.map(l => COLORS[l] || '#94a3b8');

  const canvas = $('distChart');
  if (distChart) distChart.destroy();
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  distChart = new Chart(ctx, {
    type: 'doughnut',
    data: { labels, datasets: [{ data: values, backgroundColor: colors.map(c => c + 'cc'), borderColor: colors, borderWidth: 2, hoverBorderWidth: 3 }] },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: '68%',
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: '#172119', borderColor: 'rgba(255,255,255,0.14)', borderWidth: 1, titleColor: '#fffffb', bodyColor: '#dfe8d6',
          callbacks: { label: ctx => ` ${ctx.label}: ${ctx.raw} análise(s)` } },
      },
    },
  });

  // Legenda customizada
  const total = values.reduce((a, b) => a + b, 0);
  legend.innerHTML = distribution.map((d, i) => `
    <div class="dist-item">
      <span class="dist-dot" style="background:${colors[i]}"></span>
      <span class="dist-name">${d.faixa}</span>
      <span class="dist-count">${d.qtd} (${Math.round(d.qtd/total*100)}%)</span>
    </div>`).join('');
}

function renderChart(data) {
  const empty = $('chartEmpty');
  const subtitle = $('chartSubtitle');
  if (!data.length) {
    empty.hidden = false;
    if (temporalChart) { temporalChart.destroy(); temporalChart = null; }
    return;
  }
  empty.hidden = true;
  if (data.length === 1) {
    subtitle.textContent = `1 ponto registrado, adicione mais rótulos de dia para ver a evolução`;
  } else {
    subtitle.textContent = `${data.length} pontos temporais, ${data[0].day} a ${data[data.length-1].day}`;
  }

  Chart.defaults.color = '#7b8879';
  Chart.defaults.font.family = "'IBM Plex Sans', sans-serif";
  const canvas = $('temporalChart');
  const ctx = canvas.getContext('2d');
  if (temporalChart) temporalChart.destroy();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  temporalChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.map(d => d.day),
      datasets: [
        { label: 'Taxa de Germinação (%)', data: data.map(d => +d.avg_germination_rate.toFixed(1)),
          borderColor: '#2f6f4e', backgroundColor: 'rgba(47,111,78,0.10)', pointBackgroundColor: '#2f6f4e',
          pointRadius: 6, pointHoverRadius: 9, borderWidth: 2.5, tension: 0.3, fill: true, yAxisID: 'y' },
        { label: 'Folhas médias/muda', data: data.map(d => +d.avg_leaf_count.toFixed(1)),
          borderColor: '#b77433', backgroundColor: 'rgba(183,116,51,0.08)', pointBackgroundColor: '#b77433',
          pointRadius: 6, pointHoverRadius: 9, borderWidth: 2.5, tension: 0.3, fill: true, yAxisID: 'y1' },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: '#4f5f54', usePointStyle: true, pointStyleWidth: 10 } },
        tooltip: { backgroundColor: '#172119', borderColor: 'rgba(255,255,255,0.14)', borderWidth: 1, titleColor: '#fffffb', bodyColor: '#dfe8d6', padding: 10,
          callbacks: {
            label: ctx => ctx.datasetIndex === 0 ? ` Germinação: ${ctx.raw}%` : ` Folhas: ${ctx.raw}`,
            afterBody: (items) => { const pt = data[items[0].dataIndex]; return pt ? [`  Análises: ${pt.num_analyses}`] : []; }
          }
        },
        annotation: data.length > 1 ? {} : undefined,
      },
      scales: {
        x: { grid: { color: 'rgba(23,33,25,0.06)' }, ticks: { color: '#7b8879' } },
        y:  { type:'linear', position:'left',  min:0, max:100, grid:{ color:'rgba(23,33,25,0.06)' }, ticks:{ color:'#2f6f4e', callback: v => v+'%' }, title:{ display:true, text:'Germinação (%)', color:'#2f6f4e', font:{size:11} } },
        y1: { type:'linear', position:'right', min:0, grid:{ drawOnChartArea:false }, ticks:{ color:'#b77433' }, title:{ display:true, text:'Folhas médias', color:'#b77433', font:{size:11} } },
      },
    },
  });
}
$('refreshChart').addEventListener('click', () => { loadHistory(); loadTemporal(); showToast('Atualizado.','success'); });

// ── Chatbot ───────────────────────────────────────────────────────────────────
const chatPanel    = $('chatPanel');
const chatMessages = $('chatMessages');
const chatInput    = $('chatInput');

function toggleChat() {
  const open = !chatPanel.hidden;
  chatPanel.hidden = open;
  $('chatFabIcon').textContent = open ? 'GV' : '×';
  $('chatUnread').hidden = true;
  if (!open) setTimeout(() => chatInput.focus(), 200);
}
$('chatFab').addEventListener('click', toggleChat);
$('chatClose').addEventListener('click', toggleChat);

document.querySelectorAll('.chip').forEach(btn => {
  btn.addEventListener('click', () => { chatInput.value = btn.dataset.msg; sendChat(); });
});

chatInput.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); } });
$('chatSend').addEventListener('click', sendChat);

function addMsg(text, role, animate = false) {
  const wrap = document.createElement('div');
  wrap.className = `chat-msg ${role} msg-enter`;
  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble';
  const time = document.createElement('span');
  time.className = 'chat-time';
  time.textContent = now();
  wrap.appendChild(bubble);
  wrap.appendChild(time);
  chatMessages.appendChild(wrap);

  if (animate && role === 'bot') {
    typewriter(bubble, text);
  } else {
    bubble.innerHTML = formatMd(text);
  }

  requestAnimationFrame(() => {
    wrap.classList.add('msg-visible');
    chatMessages.scrollTop = chatMessages.scrollHeight;
  });
  return wrap;
}

function typewriter(el, rawText, speed = 18) {
  const html = formatMd(rawText);
  // Para textos longos, revela por chunks; para curtos, letra a letra
  el.innerHTML = '';
  const temp = document.createElement('div');
  temp.innerHTML = html;
  const fullText = temp.innerHTML;

  let i = 0;
  let tag = false; // dentro de tag HTML
  const delay = Math.max(speed - Math.floor(rawText.length / 30), 6);

  function tick() {
    if (i >= fullText.length) {
      el.innerHTML = fullText;
      chatMessages.scrollTop = chatMessages.scrollHeight;
      return;
    }
    // Pula tags HTML inteiras (não anima dentro de <strong> etc.)
    if (fullText[i] === '<') tag = true;
    if (tag) {
      const end = fullText.indexOf('>', i);
      i = end + 1;
      tag = false;
      el.innerHTML = fullText.slice(0, i) + '<span class="cursor-blink">▋</span>';
    } else {
      i++;
      el.innerHTML = fullText.slice(0, i) + '<span class="cursor-blink">▋</span>';
    }
    chatMessages.scrollTop = chatMessages.scrollHeight;
    setTimeout(tick, delay);
  }
  tick();
}

function addTyping() {
  const wrap = document.createElement('div');
  wrap.className = 'chat-msg bot msg-enter';
  wrap.innerHTML = '<div class="chat-typing"><span></span><span></span><span></span></div>';
  chatMessages.appendChild(wrap);
  requestAnimationFrame(() => wrap.classList.add('msg-visible'));
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return wrap;
}

function formatMd(text) {
  if (typeof marked !== 'undefined') {
    return marked.parse(text);
  }
  // Fallback
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/`(.*?)`/g, '<code>$1</code>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/\n\n/g, '<br><br>')
    .replace(/\n/g, '<br>');
}

async function sendChat() {
  const msg = chatInput.value.trim();
  if (!msg) return;
  chatInput.value = '';
  $('chatSuggestions').hidden = true;
  $('chatSend').disabled = true;
  chatInput.disabled = true;

  addMsg(msg, 'user');

  // Delay mínimo de digitação: simula leitura + resposta
  const thinkMs = 600 + Math.random() * 600;
  const typing = addTyping();

  try {
    const [res] = await Promise.all([
      fetch('/api/chat', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ message: msg }) }),
      new Promise(r => setTimeout(r, thinkMs)),  // garante tempo mínimo
    ]);
    const data = await res.json();
    typing.remove();
    const reply = data.reply || data.error || 'Algo deu errado 😕';
    addMsg(reply, 'bot', true);  // animate = true → typewriter
  } catch {
    typing.remove();
    addMsg('Erro ao conectar com o servidor. Verifique se o Flask está rodando.', 'bot', false);
  } finally {
    $('chatSend').disabled = false;
    chatInput.disabled = false;
    chatInput.focus();
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────
(async () => {
  await checkStatus();
  await loadHistory(true);
  await loadTemporal();
  const loadMoreBtn = $('loadMore');
  if (loadMoreBtn) loadMoreBtn.addEventListener('click', loadMoreHistory);
  // Mostra badge do chat após 3s
  setTimeout(() => { if (chatPanel.hidden) $('chatUnread').hidden = false; }, 3000);
})();
