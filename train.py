"""
Treino local do detector de mudas — roda no Mac com Metal (MPS).
Uso: python train.py
O modelo treinado sera salvo em runs/train/train_<timestamp>/weights/best.pt.
"""
from pathlib import Path
from datetime import datetime
import yaml
import torch

# ── Caminhos ──────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent
DATASET    = BASE / "dataset"
DATA_YAML  = DATASET / "data.yaml"
FIXED_YAML = BASE / "data_train.yaml"
MODEL_DIR  = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)
RUN_NAME   = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("🍎  Metal (MPS) disponível — usando GPU do Mac")
elif torch.cuda.is_available():
    DEVICE = 0
    print("🟢  CUDA disponível")
else:
    DEVICE = "cpu"
    print("⚠️  Usando CPU — vai demorar mais, mas funciona")

# ── Corrige caminhos do data.yaml ─────────────────────────────────────────────
with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)

def resolve(raw: str, fallback: str) -> str:
    p = Path(raw)
    if not p.is_absolute():
        p = (DATASET / raw).resolve()
    if p.exists():
        return str(p)
    alt = (DATASET / fallback).resolve()
    return str(alt) if alt.exists() else raw

cfg["train"] = resolve(cfg.get("train", "train/images"), "train/images")
cfg["val"]   = resolve(cfg.get("val",   "valid/images"), "valid/images")
cfg["test"]  = resolve(cfg.get("test",  "test/images"),  "test/images")

with open(FIXED_YAML, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"\n📂  Dataset: {DATASET.name}")
print(f"   Classes : {cfg['names']}")
print(f"   Train   : {cfg['train']}")
print(f"   Val     : {cfg['val']}")
print(f"   Device  : {DEVICE}\n")

# ── Treino ────────────────────────────────────────────────────────────────────
from ultralytics import YOLO

model = YOLO(str(BASE / "yolo11s.pt"))
print("Iniciando treino morango_v3 a partir de yolo11s.pt (base limpa)…")

results = model.train(
    data=str(FIXED_YAML),
    epochs=100,
    imgsz=1280,
    batch=8,
    patience=20,
    device=DEVICE,
    workers=0,
    project=str(BASE / "runs" / "train"),
    name=RUN_NAME,
    exist_ok=False,
    cache=False,
    resume=False,
    # HSV: mantido para lidar com LED roxo, levemente suavizado
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    # geométrico: suavizado — câmera top-down, sem perspectiva real
    degrees=5.0,
    translate=0.1,
    scale=0.3,
    shear=0.0,
    perspective=0.0,
    flipud=0.3,
    fliplr=0.5,
    # mistura: mosaic parcial, mixup e copy_paste desativados
    mosaic=0.5,
    mixup=0.0,
    copy_paste=0.0,
)

# ── Resultado ─────────────────────────────────────────────────────────────────
best_src = Path(results.save_dir) / "weights" / "best.pt"

if best_src.exists():
    print(f"\n✅  Modelo treinado salvo em: {best_src}")
    print("   models/best.pt NAO foi substituido automaticamente.")
    print("   Valide o modelo novo antes de instalar.")
else:
    print(f"\n⚠️  Não encontrou best.pt em {best_src}")
    print(f"   Procure manualmente em: {BASE / 'runs'}")
