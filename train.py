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

BEST_PT = BASE / "models" / "best.pt"
if BEST_PT.exists():
    model = YOLO(str(BEST_PT))
    print(f"🔁  Iniciando fine-tune a partir de: {BEST_PT}")
else:
    model = YOLO("yolo11s.pt")

print("🚀  Iniciando treino…")
results = model.train(
    data=str(FIXED_YAML),
    epochs=100,
    imgsz=896,
    batch=2,
    patience=15,
    device=DEVICE,
    workers=0,
    project=str(BASE / "runs"),
    name="morango_v2",
    exist_ok=True,
    cache=False,
    resume=False,
    # augmentation para luz LED grow (amarelo/verde/azul/vermelho)
    hsv_h=0.5,
    hsv_s=0.9,
    hsv_v=0.5,
    # augmentation geométrico para câmera torta e plantas distantes
    degrees=15.0,
    translate=0.2,
    scale=0.5,
    shear=5.0,
    perspective=0.001,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.3,
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
