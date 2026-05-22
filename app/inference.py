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


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _bbox_intersection(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _is_duplicate_germination(
    candidate: tuple[int, int, int, int],
    kept: tuple[int, int, int, int],
) -> bool:
    inter = _bbox_intersection(candidate, kept)
    if inter <= 0:
        return False
    area_candidate = _bbox_area(candidate)
    area_kept = _bbox_area(kept)
    union = area_candidate + area_kept - inter
    iou = inter / max(union, 1)
    overlap_smaller = inter / max(min(area_candidate, area_kept), 1)
    return iou >= 0.35 or overlap_smaller >= 0.60


def _dedupe_germination_boxes(
    boxes: list[tuple[str, float, tuple[int, int, int, int]]],
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """Remove duplicatas da classe Germinacao preservando a box de maior confiança."""
    germ = [box for box in boxes if box[0] in GERMINATION_CLASSES]
    others = [box for box in boxes if box[0] not in GERMINATION_CLASSES]
    kept: list[tuple[str, float, tuple[int, int, int, int]]] = []

    for item in sorted(germ, key=lambda x: x[1], reverse=True):
        _, _, bbox = item
        if any(_is_duplicate_germination(bbox, kept_item[2]) for kept_item in kept):
            continue
        kept.append(item)

    removed = len(germ) - len(kept)
    if removed:
        print(f"  [germ] {removed} duplicata(s) de germinação removida(s)")
    return kept + others


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    w: int,
    h: int,
    ratio: float = 0.04,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    pad_x = int((x2 - x1) * ratio)
    pad_y = int((y2 - y1) * ratio)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(w, x2 + pad_x),
        min(h, y2 + pad_y),
    )


def _leaf_based_germination_fallback(
    boxes: list[tuple[str, float, tuple[int, int, int, int]]],
    w: int,
    h: int,
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """Promove folha grande sem planta associada para germinação provável."""
    germ_boxes = [bbox for cls_name, _, bbox in boxes if cls_name in GERMINATION_CLASSES]
    additions: list[tuple[str, float, tuple[int, int, int, int]]] = []
    min_area = max(12000, int(w * h * 0.008))

    for cls_name, conf, bbox in boxes:
        if cls_name != LEAF_CLASS:
            continue
        x1, y1, x2, y2 = bbox
        box_w, box_h = x2 - x1, y2 - y1
        area = _bbox_area(bbox)
        if area < min_area or box_w < 70 or box_h < 70:
            continue
        if any(_is_duplicate_germination(bbox, germ_bbox) for germ_bbox in germ_boxes):
            continue
        if any(_is_duplicate_germination(bbox, item[2]) for item in additions):
            continue
        derived_conf = round(max(0.35, min(conf * 0.85, 0.62)), 3)
        additions.append(("Germinacao", derived_conf, _expand_bbox(bbox, w, h)))

    if additions:
        print(f"  [germ] {len(additions)} germinação provável adicionada por fallback de folhas")
    return boxes + additions


def _green_component_germination_fallback(
    img_bgr: np.ndarray,
    boxes: list[tuple[str, float, tuple[int, int, int, int]]],
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """Promove agrupamentos verdes grandes que o YOLO viu como planta/folha parcial."""
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (25, 35, 45), (95, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    germ_boxes = [bbox for cls_name, _, bbox in boxes if cls_name in GERMINATION_CLASSES]
    additions: list[tuple[str, float, tuple[int, int, int, int]]] = []
    min_area = max(4500, int(w * h * 0.003))

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 55 or bh < 85:
            continue
        fill_ratio = area / max(bw * bh, 1)
        if fill_ratio < 0.12:
            continue

        bbox = _expand_bbox((x, y, x + bw, y + bh), w, h, ratio=0.10)
        if any(_is_duplicate_germination(bbox, germ_bbox) for germ_bbox in germ_boxes):
            continue
        if any(_is_duplicate_germination(bbox, item[2]) for item in additions):
            continue
        additions.append(("Germinacao", 0.5, bbox))

    if additions:
        print(f"  [germ] {len(additions)} germinação provável adicionada por máscara verde")
    return boxes + additions


def _filter_germination_by_green_signal(
    img_bgr: np.ndarray,
    boxes: list[tuple[str, float, tuple[int, int, int, int]]],
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """Descarta boxes de germinação sem tecido vegetal visível."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    kept: list[tuple[str, float, tuple[int, int, int, int]]] = []
    removed = 0

    for item in boxes:
        cls_name, _, bbox = item
        if cls_name not in GERMINATION_CLASSES:
            kept.append(item)
            continue

        x1, y1, x2, y2 = bbox
        area = _bbox_area(bbox)
        crop = hsv[y1:y2, x1:x2]
        if crop.size == 0 or area <= 0:
            removed += 1
            continue
        mask = cv2.inRange(crop, (25, 35, 45), (95, 255, 255))
        green_ratio = cv2.countNonZero(mask) / max(area, 1)
        if green_ratio < 0.04:
            removed += 1
            continue
        kept.append(item)

    if removed:
        print(f"  [germ] {removed} box(es) sem sinal verde removida(s)")
    return kept


def _split_tall_germination_boxes(
    img_bgr: np.ndarray,
    boxes: list[tuple[str, float, tuple[int, int, int, int]]],
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """Divide uma box alta quando ela cobre duas mudas separadas."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w = img_bgr.shape[:2]
    result: list[tuple[str, float, tuple[int, int, int, int]]] = []
    splits = 0

    for item in boxes:
        cls_name, conf, bbox = item
        if cls_name not in GERMINATION_CLASSES:
            result.append(item)
            continue

        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0 or bh / max(bw, 1) < 2.2 or _bbox_area(bbox) < 25000:
            result.append(item)
            continue

        crop = hsv[y1:y2, x1:x2]
        mask = cv2.inRange(crop, (25, 35, 45), (95, 255, 255))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        components: list[tuple[int, int, int, int]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < max(1800, _bbox_area(bbox) * 0.06):
                continue
            cx, cy, cw, ch = cv2.boundingRect(contour)
            if cw < 45 or ch < 45:
                continue
            components.append((x1 + cx, y1 + cy, x1 + cx + cw, y1 + cy + ch))

        components = sorted(components, key=lambda b: (b[1] + b[3]) / 2)
        if len(components) < 2:
            result.append(item)
            continue

        separated = [
            components[0],
            *[
                comp for comp in components[1:]
                if ((comp[1] + comp[3]) / 2) - ((components[0][1] + components[0][3]) / 2) > bh * 0.22
            ],
        ]
        if len(separated) < 2:
            result.append(item)
            continue

        splits += 1
        for comp in separated:
            result.append((cls_name, round(max(0.45, min(conf, 0.72)), 3), _expand_bbox(comp, w, h, ratio=0.12)))

    if splits:
        print(f"  [germ] {splits} box(es) alta(s) dividida(s) por componentes verdes")
    return result


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
    'caption', 'detected_visible', 'fallback_default'.
    """
    global _cell_detection_stats

    # 1. Caption tem prioridade absoluta
    if tray_capacity_override is not None:
        _cell_detection_stats["success"] += 1
        return tray_capacity_override, "caption"

    # 2. Valida detecção automática de células visíveis no enquadramento.
    # Se a contagem empata com as plantas, o algoritmo provavelmente não viu
    # células vazias. Nesse caso a taxa automática vira "não confirmada".
    if raw_detected is not None:
        min_plausible = max(2, germinated_count + 1)
        if min_plausible <= raw_detected <= 500:
            _cell_detection_stats["success"] += 1
            return raw_detected, "detected_visible"
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
    # SAHI ajuda em fotos grandes, mas em recortes pequenos (ex: 512x512)
    # tende a gerar caixas largas atravessando várias células.
    use_sahi = _SAHI_AVAILABLE and min(h, w) >= 640
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

    # Primeiro passe: clamp nas bordas e dedupe de boxes de germinação.
    clamped_boxes: list[tuple[str, float, tuple[int, int, int, int]]] = []

    for d in raw_boxes:
        cls_name = d["cls_name"]
        conf = d["conf"]
        x1, y1, x2, y2 = d["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bbox = (x1, y1, x2, y2)
        clamped_boxes.append((cls_name, conf, bbox))

    clamped_boxes = _dedupe_germination_boxes(clamped_boxes)
    clamped_boxes = _leaf_based_germination_fallback(clamped_boxes, w, h)
    clamped_boxes = _green_component_germination_fallback(img_bgr, clamped_boxes)
    clamped_boxes = _filter_germination_by_green_signal(img_bgr, clamped_boxes)
    clamped_boxes = _split_tall_germination_boxes(img_bgr, clamped_boxes)
    clamped_boxes = _dedupe_germination_boxes(clamped_boxes)
    germ_boxes: list[tuple[int, int, int, int, float]] = [
        (bbox[0], bbox[1], bbox[2], bbox[3], conf)
        for cls_name, conf, bbox in clamped_boxes
        if cls_name in GERMINATION_CLASSES
    ]
    folha_boxes: list[tuple[int, int, int, int]] = [
        bbox
        for cls_name, _, bbox in clamped_boxes
        if cls_name == LEAF_CLASS
    ]

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
    label_scale = max(0.38, min(w, h) / 1800)
    label_thickness = max(1, int(label_scale * 2))
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
            label = f"#{plant_id}"
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
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_thickness)
        ly = max(y1 - 6, lh + 4)
        cv2.rectangle(img_annotated, (x1, ly - lh - 4), (x1 + lw + 4, ly + baseline), color_bgr, -1)
        cv2.putText(img_annotated, label, (x1 + 2, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255, 255, 255), label_thickness, cv2.LINE_AA)

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
        "_💡 Não consegui contar as células visíveis com segurança. Para taxa confiável, "
        "envie a bandeja inteira ou informe o total de células na legenda (ex: '128')._"
        if cells_origin == "fallback_default"
        else None
    )

    germination_rate = round(germinated_count / detected_cells * 100, 1) if detected_cells > 0 else 0.0
    leaf_avg = round(sum(leaf_counts) / len(leaf_counts), 1) if leaf_counts else 0.0
    elapsed = round(time.time() - t0, 2)
    rate_scope = {
        "caption": "tray",
        "detected_visible": "visible_area",
        "fallback_default": "default_capacity",
    }.get(cells_origin, "visible_area")

    return {
        "total_detected":   total,
        "germinated":       germinated_count,
        "germination_rate": germination_rate,
        "cells_detected":   detected_cells,
        "cells_origin":     cells_origin,
        "cells_warning":    cells_warning,
        "rate_reliable":    cells_origin != "fallback_default",
        "rate_scope":       rate_scope,
        "leaf_avg":         leaf_avg,
        "total_folhas_estimadas": int(round(leaf_avg * germinated_count)),
        "leaf_counts":      leaf_counts,
        "tray_capacity":    TRAY_CAPACITY,
        "detections":       detections,
        "result_image":     f"/static/results/{result_name}",
        "inference_time_s": elapsed,
    }
