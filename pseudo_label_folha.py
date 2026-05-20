"""
Pseudo-labeling de Folha (class 1) nas imagens rf_ do dataset.
Usa segmentação HSV para detectar regiões verdes não cobertas por bboxes de Germinacao.
"""
import cv2
import numpy as np
import os
import glob
import shutil
from pathlib import Path

DATASET_DIR = Path("/Users/nikolas/projetogerminação/dataset")
BACKUP_DIR  = DATASET_DIR / "labels_backup"

# HSV verde permissivo (cotilédone + folha sob luz natural/LED fria)
HSV_LOW  = np.array([30, 35, 35])
HSV_HIGH = np.array([90, 255, 230])

# Área mínima do contorno: 0.5% da área total da imagem
MIN_AREA_FRACTION = 0.005

# IoU máximo com bbox Germinacao existente para aceitar como Folha nova
MAX_IOU = 0.3


def yolo_to_xyxy(cx: float, cy: float, bw: float, bh: float, W: int, H: int):
    x1 = int((cx - bw / 2) * W)
    y1 = int((cy - bh / 2) * H)
    x2 = int((cx + bw / 2) * W)
    y2 = int((cy + bh / 2) * H)
    return max(0, x1), max(0, y1), min(W, x2), min(H, y2)


def bbox_iou(a, b):
    """IoU entre dois bboxes (x1,y1,x2,y2)."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def process_image(img_path: Path, lbl_path: Path) -> int:
    """
    Processa uma imagem rf_, adiciona anotações de Folha se encontrar
    regiões verdes fora das bboxes existentes.
    Retorna número de anotações de Folha adicionadas.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    H, W = img.shape[:2]
    min_area = MIN_AREA_FRACTION * H * W

    # Ler labels existentes (todas Germinacao)
    existing_bboxes = []
    existing_lines = []
    if lbl_path.exists():
        with open(lbl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                existing_lines.append(line)
                parts = line.split()
                if len(parts) == 5:
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    existing_bboxes.append(yolo_to_xyxy(cx, cy, bw, bh, W, H))

    # Segmentação HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)

    # Morfologia para limpar ruído
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    new_annotations = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        x2, y2 = x + bw, y + bh
        candidate = (x, y, x2, y2)

        # Verificar IoU com bboxes existentes
        max_iou = max((bbox_iou(candidate, ex) for ex in existing_bboxes), default=0.0)
        if max_iou >= MAX_IOU:
            continue

        # Converter para formato YOLO normalizado
        cx_n = (x + bw / 2) / W
        cy_n = (y + bh / 2) / H
        bw_n = bw / W
        bh_n = bh / H

        # Descartar bboxes que cubram > 80% da imagem (falsos positivos de fundo)
        if bw_n > 0.8 and bh_n > 0.8:
            continue

        new_annotations.append(f"1 {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}")

    if new_annotations:
        with open(lbl_path, "w") as f:
            for line in existing_lines:
                f.write(line + "\n")
            for ann in new_annotations:
                f.write(ann + "\n")

    return len(new_annotations)


def run():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    splits = [
        (DATASET_DIR / "train" / "images", DATASET_DIR / "train" / "labels"),
        (DATASET_DIR / "valid" / "images", DATASET_DIR / "valid" / "labels"),
        (DATASET_DIR / "test"  / "images", DATASET_DIR / "test"  / "labels"),
    ]

    total_images = 0
    total_images_modified = 0
    total_annotations_added = 0
    sample_paths = []

    for img_dir, lbl_dir in splits:
        rf_images = list(img_dir.glob("rf_*.jpg")) + list(img_dir.glob("rf_*.jpeg"))
        for img_path in rf_images:
            stem = img_path.stem
            lbl_path = lbl_dir / f"{stem}.txt"

            # Backup antes de modificar
            if lbl_path.exists():
                backup_path = BACKUP_DIR / f"{stem}.txt"
                shutil.copy2(lbl_path, backup_path)

            total_images += 1
            added = process_image(img_path, lbl_path)
            if added > 0:
                total_images_modified += 1
                total_annotations_added += added
                if len(sample_paths) < 5:
                    sample_paths.append((img_path, lbl_path, added))

    print(f"\n=== RESULTADO ===")
    print(f"Imagens rf_ processadas: {total_images}")
    print(f"Imagens com Folha adicionada: {total_images_modified}")
    print(f"Anotações de Folha adicionadas: {total_annotations_added}")
    print(f"Backups em: {BACKUP_DIR}")

    if sample_paths:
        print(f"\n=== AMOSTRA (5 imagens modificadas) ===")
        for img_path, lbl_path, added in sample_paths:
            print(f"\nIMG: {img_path}")
            print(f"LBL: {lbl_path}")
            print(f"     +{added} anotações de Folha")
            with open(lbl_path) as f:
                for line in f:
                    print(f"     {line.rstrip()}")


if __name__ == "__main__":
    run()
