"""
Treino local do detector de mudas — roda no Mac com Metal (MPS).
Uso: python train.py
O modelo treinado será salvo em models/best.pt automaticamente.
"""
from pathlib import Path
import shutil
import yaml
import torch

# ── Caminhos ──────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent
DATASET    = BASE / "dataset"
DATA_YAML  = DATASET / "data.yaml"
FIXED_YAML = BASE / "data_train.yaml"
MODEL_DIR  = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

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
    epochs=60,
    imgsz=640,
    batch=4,
    patience=12,
    device=DEVICE,
    workers=0,
    project=str(BASE / "runs"),
    name="morango_v3",
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

# ── Copia best.pt para models/ ────────────────────────────────────────────────
best_src = Path(results.save_dir) / "weights" / "best.pt"
best_dst = MODEL_DIR / "best.pt"

if best_src.exists():
    shutil.copy(best_src, best_dst)
    print(f"\n✅  Modelo salvo em: {best_dst}")
    print("   Reinicie a app (python run.py) para usar o modelo treinado.")
else:
    print(f"\n⚠️  Não encontrou best.pt em {best_src}")
    print(f"   Procure manualmente em: {BASE / 'runs'}")
