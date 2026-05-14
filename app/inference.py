"""Pipeline de inferência: detecção de germinação + contagem de folhas por planta."""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    _SAHI_AVAILABLE = True
except ImportError:
    _SAHI_AVAILABLE = False

_SAHI_MODEL = None  # lazy init


# ── Config de classes ────────────────────────────────────────────────────────
# Dataset morango v2: 2 classes (Germinacao = planta germinada, Folha = folha individual)
CLASS_COLORS = {
    "Germinacao": (52, 211, 153),   # verde esmeralda
    "Folha":      (251, 191, 36),   # amarelo
}
DEFAULT_COLOR = (148, 163, 184)

GERMINATION_CLASSES = {"Germinacao"}
LEAF_CLASS = "Folha"

# Capacidade da bandeja (denominador da taxa de germinação). Padrão: 200 células (bandeja morango).
TRAY_CAPACITY = int(os.environ.get("TRAY_CAPACITY", "200"))


def load_model(model_path: str):
    """Carrega YOLO11. Usa best.pt se disponível, senão fallback COCO."""
    from ultralytics import YOLO

    p = Path(model_path)
    if p.exists():
        print(f"  ✅ Modelo carregado: {p}")
        return YOLO(str(p))
    print("  ⚠️  best.pt não encontrado — usando yolo11s.pt (COCO pré-treinado)")
    print("     Para usar seu modelo treinado: coloque best.pt em models/")
    return YOLO("yolo11s.pt")


def _get_sahi_model(model_path: str):
    global _SAHI_MODEL
    if _SAHI_MODEL is None and _SAHI_AVAILABLE:
        _SAHI_MODEL = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=0.20,
            device="cpu",
        )
    return _SAHI_MODEL


def _run_inference_sahi(
    img_path: Path,
    model_path: str,
    names: dict,
) -> list[dict]:
    sahi_model = _get_sahi_model(model_path)
    sliced = get_sliced_prediction(
        image=str(img_path),
        detection_model=sahi_model,
        slice_height=768,
        slice_width=1024,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        postprocess_match_threshold=0.5,
        verbose=0,
    )
    detections = []
    for pred in sliced.object_prediction_list:
        x1 = int(pred.bbox.minx)
        y1 = int(pred.bbox.miny)
        x2 = int(pred.bbox.maxx)
        y2 = int(pred.bbox.maxy)
        cls_name = pred.category.name
        detections.append({
            "cls_name": cls_name,
            "conf": float(pred.score.value),
            "bbox": (x1, y1, x2, y2),
        })
    return detections


def _leaves_inside(folha_boxes: list[tuple[int, int, int, int]], germ_bbox: tuple[int, int, int, int]) -> int:
    """Conta quantas bbox de Folha têm centro dentro da bbox de Germinacao."""
    x1, y1, x2, y2 = germ_bbox
    n = 0
    for fx1, fy1, fx2, fy2 in folha_boxes:
        cx = (fx1 + fx2) / 2
        cy = (fy1 + fy2) / 2
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            n += 1
    return n


def _estimate_leaves_by_contours(crop_bgr: np.ndarray) -> int:
    """Fallback: estima # de folhas por análise de contornos verdes no crop."""
    if crop_bgr is None or crop_bgr.size == 0:
        return 0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    # Verde morango: matiz 25-85 cobre verde claro a verde escuro
    mask = cv2.inRange(hsv, (25, 30, 30), (85, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (crop_bgr.shape[0] * crop_bgr.shape[1]) * 0.005
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    return max(1, len(valid)) if valid else 0


def run_inference(
    image_path: str,
    model,
    result_folder: str,
    conf_threshold: float = 0.25,
    class_names: Optional[list[str]] = None,
) -> dict:
    """Roda detecção YOLO e retorna métricas + imagem anotada."""
    t0 = time.time()
    img_path = Path(image_path)

    # Lê imagem antes para decidir estratégia
    img_array = np.fromfile(str(img_path), dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        pil = Image.open(img_path).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    h, w = img_bgr.shape[:2]
    use_sahi = _SAHI_AVAILABLE and min(h, w) >= 900
    print(f"  Inferencia {'SAHI (tiles)' if use_sahi else 'direta'} em {w}x{h}")

    # raw_boxes: lista uniforme de dicts {cls_name, conf, bbox}
    raw_boxes: list[dict] = []

    if use_sahi:
        model_path = str(getattr(model, "ckpt_path", "") or "models/best.pt")
        fallback_names = {0: "Germinacao", 1: "Folha"}
        names = class_names or fallback_names
        raw_boxes = _run_inference_sahi(img_path, model_path, names)
        # Filtro class-aware: Folha aceita conf 0.10 abaixo do threshold
        folha_conf = max(0.15, conf_threshold - 0.10)
        raw_boxes = [
            d for d in raw_boxes
            if (d["cls_name"] == "Germinacao" and d["conf"] >= conf_threshold)
            or (d["cls_name"] == "Folha" and d["conf"] >= folha_conf)
            or (d["cls_name"] not in ("Germinacao", "Folha") and d["conf"] >= conf_threshold)
        ]
    else:
        # Usa conf mais baixo no predict para capturar Folhas; filtra class-aware depois
        folha_conf = max(0.15, conf_threshold - 0.10)
        results = model.predict(
            source=str(img_path),
            conf=folha_conf,
            imgsz=1280,
            verbose=False,
        )
        result = results[0]
        names = class_names or result.names
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = names[cls_id] if isinstance(names, dict) else str(cls_id)
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            raw_boxes.append({"cls_name": cls_name, "conf": conf, "bbox": (x1, y1, x2, y2)})
        # Filtro class-aware: Germinacao precisa de conf_threshold normal
        raw_boxes = [
            d for d in raw_boxes
            if (d["cls_name"] == "Germinacao" and d["conf"] >= conf_threshold)
            or (d["cls_name"] == "Folha" and d["conf"] >= folha_conf)
            or (d["cls_name"] not in ("Germinacao", "Folha") and d["conf"] >= conf_threshold)
        ]

    img_annotated = img_bgr.copy()

    # Primeiro passe: separa boxes por classe (clamp nas bordas)
    germ_boxes: list[tuple[int, int, int, int, float]] = []
    folha_boxes: list[tuple[int, int, int, int]] = []
    clamped_boxes: list[tuple[str, float, tuple[int, int, int, int]]] = []

    for d in raw_boxes:
        cls_name = d["cls_name"]
        conf = d["conf"]
        x1, y1, x2, y2 = d["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bbox = (x1, y1, x2, y2)
        clamped_boxes.append((cls_name, conf, bbox))

        if cls_name in GERMINATION_CLASSES:
            germ_boxes.append((x1, y1, x2, y2, conf))
        elif cls_name == LEAF_CLASS:
            folha_boxes.append(bbox)

    raw_boxes_for_loop = clamped_boxes

    # Segundo passe: anota e calcula folhas por germinação via containment
    leaf_counts: list[int] = []
    detections: list[dict] = []
    scale = max(0.4, min(w, h) / 1200)
    thickness = max(1, int(scale * 2))

    for cls_name, conf, bbox in raw_boxes_for_loop:
        x1, y1, x2, y2 = bbox
        color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
        color_bgr = (color[2], color[1], color[0])

        if cls_name in GERMINATION_CLASSES:
            leaf_n = _leaves_inside(folha_boxes, bbox)
            # Fallback: modelo não detectou Folhas via YOLO — estima por contornos verdes
            if leaf_n == 0:
                crop = img_bgr[y1:y2, x1:x2]
                leaf_n = _estimate_leaves_by_contours(crop)
            leaf_counts.append(leaf_n)
            label = f"Germinacao {conf:.0%} | {leaf_n} folhas"
            germinated = True
        elif cls_name == LEAF_CLASS:
            leaf_n = 1
            label = f"Folha {conf:.0%}"
            germinated = False
        else:
            leaf_n = 0
            label = f"{cls_name} {conf:.0%}"
            germinated = False

        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color_bgr, thickness)
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

    germinated_count = len(germ_boxes)
    leaves_total = len(folha_boxes)
    total = len(detections)
    germination_rate = round(germinated_count / TRAY_CAPACITY * 100, 1) if TRAY_CAPACITY > 0 else 0.0
    leaf_avg = round(sum(leaf_counts) / len(leaf_counts), 1) if leaf_counts else 0.0
    elapsed = round(time.time() - t0, 2)

    return {
        "total_detected":   total,
        "germinated":       germinated_count,
        "germination_rate": germination_rate,
        "leaves_total":     leaves_total,
        "leaf_avg":         leaf_avg,
        "leaf_counts":      leaf_counts,
        "tray_capacity":    TRAY_CAPACITY,
        "detections":       detections,
        "result_image":     f"/static/results/{result_name}",
        "inference_time_s": elapsed,
    }
