"""Pipeline de inferência: detecção de germinação + contagem de folhas por planta."""
from __future__ import annotations

import os
import re
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


# ── Caption parsing (compartilhado entre UI web e WhatsApp) ──────────────────

_CAPTION_CAPACITY_RE = re.compile(r"\b([1-9]\d{1,2})\b")


def parse_caption(raw: str | None) -> tuple[str | None, int | None]:
    """Retorna (caption_sanitizada, capacidade_ou_None).

    Capacidade: primeiro número entre 12 e 500 na caption.
    Caption: limitada a 100 chars, sem caracteres de controle.
    """
    if not raw:
        return None, None
    caption = re.sub(r"[^\x20-\x7EÀ-ɏЀ-ӿ\n\r]", "", raw)
    caption = " ".join(caption.split())[:100].strip() or None
    capacity: int | None = None
    for m in _CAPTION_CAPACITY_RE.finditer(raw):
        n = int(m.group(1))
        if 12 <= n <= 500:
            capacity = n
            break
    return caption, capacity


# ── Normalização de iluminação ───────────────────────────────────────────────

def _normalize_lighting(img_bgr: np.ndarray) -> np.ndarray:
    """White balance (Gray World) + CLAHE no canal L para robustez a LED colorido."""
    # Gray World white balance
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    mean_b, mean_g, mean_r = b.mean(), g.mean(), r.mean()
    mean_gray = (mean_b + mean_g + mean_r) / 3.0
    if mean_b > 1 and mean_g > 1 and mean_r > 1:
        b = np.clip(b * (mean_gray / mean_b), 0, 255)
        g = np.clip(g * (mean_gray / mean_g), 0, 255)
        r = np.clip(r * (mean_gray / mean_r), 0, 255)
    balanced = cv2.merge([b, g, r]).astype(np.uint8)

    # CLAHE no canal L (Lab) para realçar contraste sem saturar cor
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2Lab)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    result = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_Lab2BGR)
    print("  Normalização de iluminação aplicada (Gray World + CLAHE)")
    return result


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
    tile_size = 416  # tile fixo, melhor para plantas pequenas/distantes
    overlap_ratio = 0.3  # overlap maior compensa tile menor
    sahi_model = _get_sahi_model(model_path)
    sliced = get_sliced_prediction(
        image=str(img_path),
        detection_model=sahi_model,
        slice_height=tile_size,
        slice_width=tile_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
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
    mask = cv2.inRange(hsv, (15, 20, 20), (95, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    crop_area = float(crop_bgr.shape[0] * crop_bgr.shape[1])
    green_ratio = float(cv2.countNonZero(mask)) / max(crop_area, 1)
    if green_ratio <= 0.02:
        return 0

    # Distance transform: threshold 0.25 (mais permissivo, captura mais peaks sobrepostos)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.25 * dist.max(), 255, 0)
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

    # Boost: quando watershed identifica muitos peaks, confiar nele diretamente
    if n_peaks >= 3:
        n_estimated = max(n_estimated, n_peaks)

    # Floor proporcional: combina tamanho absoluto do crop + densidade de verde
    # Calibrado para bboxes YOLO reais (máx ~38k px, maioria 11k-30k)
    if crop_area >= 25000 and green_ratio >= 0.40:
        n_estimated = max(n_estimated, 4)
    elif crop_area >= 12000 and green_ratio >= 0.40:
        n_estimated = max(n_estimated, 3)
    elif crop_area >= 5000 and green_ratio >= 0.30:
        n_estimated = max(n_estimated, 2)
    else:
        n_estimated = max(n_estimated, 1)

    # Cap descendente por tamanho: muda pequena não pode reportar muitas folhas
    if crop_area < 5000:
        n_estimated = min(n_estimated, 1)   # muda muito pequena: 1 cotilédone
    elif crop_area < 12000:
        n_estimated = min(n_estimated, 2)   # cotilédones: max 2
    elif crop_area < 25000:
        n_estimated = min(n_estimated, 3)   # planta jovem: max 3 folhas

    # Cap geral: morango D7-D14 com cotilédones pode ter até 6 folhas
    return min(n_estimated, 6)


_cell_detection_stats: dict[str, int] = {"success": 0, "fallback": 0}


def get_cell_detection_stats() -> dict[str, int]:
    """Retorna telemetria acumulada de detecções de células."""
    return dict(_cell_detection_stats)


def _resolve_cell_count(
    raw_detected: Optional[int],
    germinated_count: int,
    tray_capacity_override: Optional[int],
) -> tuple[int, str]:
    """Sanity check + fallback hierárquico para contagem de células.

    Retorna (células_usadas, origem) onde origem é uma das strings:
    'caption', 'detected', 'fallback_default'.
    """
    global _cell_detection_stats

    # 1. Caption tem prioridade absoluta
    if tray_capacity_override is not None:
        _cell_detection_stats["success"] += 1
        return tray_capacity_override, "caption"

    # 2. Valida detecção automática: descarta se anômala
    # mínimo = max(12, plants*3): bandeja comercial menor tem 12 células,
    # e células detectadas devem ser ao menos 3x as plantas (taxa máx ~33%)
    if raw_detected is not None:
        min_plausible = max(12, int(germinated_count * 3))
        if min_plausible <= raw_detected <= 500:
            _cell_detection_stats["success"] += 1
            return max(raw_detected, germinated_count), "detected"
        print(
            f"  [cells] Detecção anômala descartada: {raw_detected} "
            f"(germinadas={germinated_count}, mín plausível={min_plausible})"
        )

    # 3. Fallback: TRAY_CAPACITY default
    _cell_detection_stats["fallback"] += 1
    total_attempts = sum(_cell_detection_stats.values())
    print(
        f"  [cells] Fallback para TRAY_CAPACITY={TRAY_CAPACITY} "
        f"(falhas={_cell_detection_stats['fallback']}/{total_attempts})"
    )
    return max(TRAY_CAPACITY, germinated_count), "fallback_default"


def run_inference(
    image_path: str,
    model,
    result_folder: str,
    conf_threshold: float = 0.25,
    class_names: Optional[list[str]] = None,
    tray_capacity_override: Optional[int] = None,
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

    # Normaliza iluminação para inferência (original preservado para anotação visual)
    img_for_inference = _normalize_lighting(img_bgr)

    # Salva imagem normalizada em arquivo temporário para SAHI (que exige path)
    norm_path = img_path.with_name(f"_norm_{img_path.name}")
    cv2.imwrite(str(norm_path), img_for_inference)

    # raw_boxes: lista uniforme de dicts {cls_name, conf, bbox}
    raw_boxes: list[dict] = []

    try:
        if use_sahi:
            model_path = str(getattr(model, "ckpt_path", "") or "models/best.pt")
            fallback_names = {0: "Germinacao", 1: "Folha"}
            names = class_names or fallback_names
            raw_boxes = _run_inference_sahi(norm_path, model_path, names)
            # Filtro class-aware: Germinacao aceita -0.07, Folha aceita -0.10
            germ_conf = max(0.12, conf_threshold - 0.13)
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
                source=img_for_inference,
                conf=folha_conf,
                imgsz=1280,
                augment=True,
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
            germ_conf = max(0.12, conf_threshold - 0.13)
            raw_boxes = [
                d for d in raw_boxes
                if (d["cls_name"] == "Germinacao" and d["conf"] >= germ_conf)
                or (d["cls_name"] == "Folha" and d["conf"] >= folha_conf)
                or (d["cls_name"] not in ("Germinacao", "Folha") and d["conf"] >= conf_threshold)
            ]
    finally:
        # Remove arquivo temporário normalizado
        try:
            norm_path.unlink(missing_ok=True)
        except Exception:
            pass

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
            leaf_yolo = _leaves_inside(folha_boxes, bbox)
            crop = img_bgr[y1:y2, x1:x2]
            leaf_contour = _estimate_leaves_by_contours(crop)
            # Usa o maior sinal: YOLO pode subestimar (não detectou todas as Folhas),
            # contorno pode subestimar (threshold colapsou peaks sobrepostos)
            leaf_n = max(leaf_yolo, leaf_contour)
            leaf_counts.append(leaf_n)
            folhas_lbl = f"{leaf_n} folha" if leaf_n == 1 else f"{leaf_n} folhas"
            label = f"#{plant_id} Germ {conf:.0%} | {folhas_lbl}"
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

    raw_detected = _count_visible_cells(img_bgr)
    detected_cells, cells_origin = _resolve_cell_count(
        raw_detected, germinated_count, tray_capacity_override
    )
    cells_warning = (
        "⚠️ Não consegui detectar o total de células. Envie a foto com legenda "
        "contendo o número de células (ex: '128') para resultado preciso."
        if cells_origin == "fallback_default"
        else None
    )

    germination_rate = round(germinated_count / detected_cells * 100, 1) if detected_cells > 0 else 0.0
    leaf_avg = round(sum(leaf_counts) / len(leaf_counts), 1) if leaf_counts else 0.0
    elapsed = round(time.time() - t0, 2)

    return {
        "total_detected":   total,
        "germinated":       germinated_count,
        "germination_rate": germination_rate,
        "cells_detected":   detected_cells,
        "cells_origin":     cells_origin,
        "cells_warning":    cells_warning,
        "leaf_avg":         leaf_avg,
        "total_folhas_estimadas": int(round(leaf_avg * germinated_count)),
        "leaf_counts":      leaf_counts,
        "tray_capacity":    TRAY_CAPACITY,
        "detections":       detections,
        "result_image":     f"/static/results/{result_name}",
        "inference_time_s": elapsed,
    }
