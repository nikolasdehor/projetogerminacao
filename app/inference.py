"""Pipeline de inferência: detecção de mudas + heurística de contagem de folhas."""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


# ── Cores por classe ─────────────────────────────────────────────────────────
CLASS_COLORS = {
    "seedling":    (52, 211, 153),   # verde esmeralda
    "twoseedling": (16, 185, 129),   # verde escuro
    "weak":        (251, 191, 36),   # amarelo
    "noseedling":  (239, 68, 68),    # vermelho
    "processed":   (139, 92, 246),   # roxo
    "askew":       (249, 115, 22),   # laranja
}
DEFAULT_COLOR = (148, 163, 184)

# Classes que indicam germinação bem-sucedida
GERMINATION_CLASSES = {"seedling", "twoseedling", "weak", "askew", "processed"}


def load_model(model_path: str):
    """Carrega YOLO11. Usa best.pt se disponível, senão fallback COCO."""
    from ultralytics import YOLO

    p = Path(model_path)
    if p.exists():
        print(f"  ✅ Modelo carregado: {p}")
        return YOLO(str(p))
    else:
        print("  ⚠️  best.pt não encontrado — usando yolo11s.pt (COCO pré-treinado)")
        print("     Para usar seu modelo treinado: coloque best.pt em models/")
        return YOLO("yolo11s.pt")


def _estimate_leaf_count(crop_bgr: np.ndarray, class_name: str) -> int:
    """
    Heurística de contagem de folhas quando não há modelo treinado.
    Usa análise de contornos verdes na imagem recortada.
    """
    if class_name == "noseedling":
        return 0
    if class_name == "twoseedling":
        base = 4
    elif class_name == "seedling":
        base = 2
    elif class_name == "weak":
        base = 1
    else:
        base = 2

    # Tenta contar regiões verdes
    try:
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([95, 255, 255]))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 50)
        total_area = crop_bgr.shape[0] * crop_bgr.shape[1]
        ratio = green_area / max(total_area, 1)
        bonus = int(ratio * 4)
        return max(1, base + bonus) if class_name != "noseedling" else 0
    except Exception:
        return base


def run_inference(
    image_path: str,
    model,
    result_folder: str,
    conf_threshold: float = 0.25,
    class_names: Optional[list[str]] = None,
) -> dict:
    """
    Roda detecção YOLO na imagem e retorna dicionário com resultados.
    """
    t0 = time.time()
    img_path = Path(image_path)

    # Inferência
    results = model.predict(source=str(img_path), conf=conf_threshold, verbose=False)
    result = results[0]

    # Nomes das classes (usa do modelo se não passado)
    names = class_names or result.names  # dict {id: name}

    # Lê imagem original em BGR suportando caminhos unicode nativamente
    img_array = np.fromfile(str(img_path), dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        # Fallback via PIL caso o OpenCV falhe em decodificar a imagem
        pil = Image.open(img_path).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    img_annotated = img_bgr.copy()
    h, w = img_bgr.shape[:2]

    detections = []
    germinated_count = 0
    leaf_counts = []

    for box in result.boxes:
        cls_id   = int(box.cls[0])
        conf     = float(box.conf[0])
        cls_name = names[cls_id] if isinstance(names, dict) else str(cls_id)

        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Recorte para heurística de folhas
        crop = img_bgr[y1:y2, x1:x2]
        leaf_n = _estimate_leaf_count(crop, cls_name) if crop.size > 0 else 0

        germinated = cls_name in GERMINATION_CLASSES
        if germinated:
            germinated_count += 1
            leaf_counts.append(leaf_n)

        color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
        color_bgr = (color[2], color[1], color[0])

        # Escala dinâmica do texto baseada no tamanho da imagem
        scale = max(0.4, min(w, h) / 1200)
        thickness = max(1, int(scale * 2))

        # Desenha bbox
        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color_bgr, thickness)

        # Label com fundo
        label = f"{cls_name} {conf:.0%} | {leaf_n} folhas"
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        ly = max(y1 - 6, lh + 4)
        cv2.rectangle(img_annotated, (x1, ly - lh - 4), (x1 + lw + 4, ly + baseline), color_bgr, -1)
        cv2.putText(img_annotated, label, (x1 + 2, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        detections.append({
            "class":      cls_name,
            "confidence": round(conf, 3),
            "bbox":       [x1, y1, x2, y2],
            "germinated": germinated,
            "leaf_count": leaf_n,
        })

    # Salva imagem anotada
    result_name = f"result_{uuid.uuid4().hex[:16]}.jpg"
    result_path = Path(result_folder) / result_name
    cv2.imwrite(str(result_path), img_annotated)

    total = len(detections)
    germination_rate = round(germinated_count / total * 100, 1) if total > 0 else 0.0
    leaf_avg = round(sum(leaf_counts) / len(leaf_counts), 1) if leaf_counts else 0.0
    elapsed = round(time.time() - t0, 2)

    return {
        "total_detected":   total,
        "germinated":       germinated_count,
        "germination_rate": germination_rate,
        "leaf_avg":         leaf_avg,
        "leaf_counts":      leaf_counts,
        "detections":       detections,
        "result_image":     f"/static/results/{result_name}",
        "inference_time_s": elapsed,
    }
