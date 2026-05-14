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
            confidence_threshold=0.15,
            device="cpu",
        )
    return _SAHI_MODEL


def _run_inference_sahi(
    img_path: Path,
    model_path: str,
    names: dict,
) -> list[dict]:
    import cv2 as _cv2
    _img = _cv2.imread(str(img_path))
    _h, _w = _img.shape[:2] if _img is not None else (512, 512)
    # Tiles adaptivos: ~metade da menor dimensão para garantir fatiamento real
    tile = max(192, min(_h, _w) // 2)
    sahi_model = _get_sahi_model(model_path)
    sliced = get_sliced_prediction(
        image=str(img_path),
        detection_model=sahi_model,
        slice_height=tile,
        slice_width=tile,
        overlap_height_ratio=0.3,
        overlap_width_ratio=0.3,
        postprocess_match_threshold=0.4,
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


def _count_visible_cells(img_bgr: np.ndarray) -> Optional[int]:
    """Conta células visíveis da bandeja com adaptive threshold (lida com iluminação variada)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_bgr.shape[:2]
    # blockSize deve ser ímpar e maior que uma célula típica
    block = max(51, int(min(h, w) * 0.1) | 1)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block,
        C=10,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # Inverte para que células (regiões escuras) virem contornos externos
    inv = cv2.bitwise_not(closed)
    contours, _ = cv2.findContours(inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img_area = float(h * w)
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (0.005 * img_area <= area <= 0.25 * img_area):
            continue
        cw_box = cv2.boundingRect(c)[2]
        ch_box = cv2.boundingRect(c)[3]
        aspect = cw_box / max(ch_box, 1)
        if 0.4 < aspect < 2.5:
            valid.append(c)
    return len(valid) if len(valid) >= 2 else None


def _estimate_leaves_by_contours(crop_bgr: np.ndarray) -> int:
    """Estima folhas via watershed peaks + área verde. Cap 6 (morango D7-D14)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return 0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (25, 30, 30), (85, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    crop_area = float(crop_bgr.shape[0] * crop_bgr.shape[1])
    green_ratio = float(cv2.countNonZero(mask)) / max(crop_area, 1)
    if green_ratio <= 0.02:
        return 0

    # Distance transform: threshold 0.35 (mais permissivo que 0.4, menos que 0.3)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(sure_fg)
    n_peaks = max(0, num_labels - 1)  # subtrai background

    # Estimativa por área: 0.10 por folha
    # Nota: quando green_ratio > 0.40, bbox é densa e n_by_area superestima
    n_by_area = int(round(green_ratio / 0.10))

    if n_peaks > 1:
        if green_ratio > 0.40:
            # bbox densa: área não confiável — peaks é o sinal mais forte
            n_estimated = n_peaks
        else:
            # bbox esparsa: peaks pode ter colapsado, usa o maior
            n_estimated = max(n_peaks, max(1, n_by_area))
    else:
        # 0-1 peak: usa área (peaks provavelmente colapsou múltiplos)
        if green_ratio > 0.40:
            n_estimated = max(1, min(n_by_area, 4))  # teto 4 sem peaks confiáveis
        else:
            n_estimated = max(1, n_by_area)

    # Sanity check: >30% verde implica pelo menos 2 folhas
    if green_ratio >= 0.30:
        n_estimated = max(n_estimated, 2)

    # Cap: morango D7-D14 com cotilédones pode ter até 6 folhas
    return min(n_estimated, 6)


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
    use_sahi = _SAHI_AVAILABLE and min(h, w) >= 400
    print(f"  Inferencia {'SAHI (tiles)' if use_sahi else 'direta'} em {w}x{h}")

    # raw_boxes: lista uniforme de dicts {cls_name, conf, bbox}
    raw_boxes: list[dict] = []

    if use_sahi:
        model_path = str(getattr(model, "ckpt_path", "") or "models/best.pt")
        fallback_names = {0: "Germinacao", 1: "Folha"}
        names = class_names or fallback_names
        raw_boxes = _run_inference_sahi(img_path, model_path, names)
        # Filtro class-aware: Germinacao aceita -0.07, Folha aceita -0.10
        germ_conf = max(0.15, conf_threshold - 0.07)
        folha_conf = max(0.15, conf_threshold - 0.10)
        raw_boxes = [
            d for d in raw_boxes
            if (d["cls_name"] == "Germinacao" and d["conf"] >= germ_conf)
            or (d["cls_name"] == "Folha" and d["conf"] >= folha_conf)
            or (d["cls_name"] not in ("Germinacao", "Folha") and d["conf"] >= conf_threshold)
        ]
    else:
        # Usa conf mais baixo no predict para capturar Germinacoes e Folhas periféricas
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
        # Filtro class-aware: Germinacao aceita -0.07, Folha aceita -0.10
        germ_conf = max(0.15, conf_threshold - 0.07)
        raw_boxes = [
            d for d in raw_boxes
            if (d["cls_name"] == "Germinacao" and d["conf"] >= germ_conf)
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

    # Ordena clamped_boxes: Germinacao top-down, left-right (centro y depois x)
    # Folhas ficam no final (não recebem plant_id)
    def _sort_key(item):
        cls_name, conf, bbox = item
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        is_germ = 0 if cls_name in GERMINATION_CLASSES else 1
        return (is_germ, cy, cx)

    raw_boxes_for_loop = sorted(clamped_boxes, key=_sort_key)

    # Segundo passe: anota e calcula folhas por germinação via containment
    leaf_counts: list[int] = []
    detections: list[dict] = []
    scale = max(0.4, min(w, h) / 1200)
    thickness = max(1, int(scale * 2))
    plant_id = 0

    for cls_name, conf, bbox in raw_boxes_for_loop:
        x1, y1, x2, y2 = bbox
        color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
        color_bgr = (color[2], color[1], color[0])

        if cls_name in GERMINATION_CLASSES:
            plant_id += 1
            leaf_n = _leaves_inside(folha_boxes, bbox)
            # Fallback: modelo não detectou Folhas via YOLO — estima por contornos verdes
            if leaf_n == 0:
                crop = img_bgr[y1:y2, x1:x2]
                leaf_n = _estimate_leaves_by_contours(crop)
            leaf_counts.append(leaf_n)
            label = f"#{plant_id} Germ {conf:.0%} | {leaf_n}f"
            germinated = True
            pid: Optional[int] = plant_id
        elif cls_name == LEAF_CLASS:
            # Folha entra no JSON mas não é desenhada (evita poluição visual)
            detections.append({
                "class":      cls_name,
                "confidence": round(conf, 3),
                "bbox":       [x1, y1, x2, y2],
                "germinated": False,
                "leaf_count": 1,
                "plant_id":   None,
            })
            continue
        else:
            leaf_n = 0
            label = f"{cls_name} {conf:.0%}"
            germinated = False
            pid = None

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
            "plant_id":   pid,
        })

    # Salva imagem anotada
    result_name = f"result_{uuid.uuid4().hex[:16]}.jpg"
    result_path = Path(result_folder) / result_name
    cv2.imwrite(str(result_path), img_annotated)

    germinated_count = len(germ_boxes)
    leaves_total = len(folha_boxes)
    total = len(detections)
    detected_cells = _count_visible_cells(img_bgr) or TRAY_CAPACITY
    detected_cells = max(detected_cells, germinated_count)
    germination_rate = round(germinated_count / detected_cells * 100, 1) if detected_cells > 0 else 0.0
    leaf_avg = round(sum(leaf_counts) / len(leaf_counts), 1) if leaf_counts else 0.0
    elapsed = round(time.time() - t0, 2)

    return {
        "total_detected":   total,
        "germinated":       germinated_count,
        "germination_rate": germination_rate,
        "cells_detected":   detected_cells,
        "leaf_avg":         leaf_avg,
        "total_folhas_estimadas": int(round(leaf_avg * germinated_count)),
        "leaf_counts":      leaf_counts,
        "tray_capacity":    TRAY_CAPACITY,
        "detections":       detections,
        "result_image":     f"/static/results/{result_name}",
        "inference_time_s": elapsed,
    }
