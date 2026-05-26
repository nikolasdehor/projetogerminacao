"""Pipeline de inferência: detecção de germinação + contagem de folhas por planta."""
from __future__ import annotations

import os
import re
import time
import uuid
import math
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


def _reconstruct_magenta_for_yolo(img_bgr: np.ndarray) -> np.ndarray:
    """Reconstrói contraste de planta em LED magenta extremo para o YOLO."""
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    intensity = 0.35 * b + 0.15 * g + 0.50 * r
    intensity = cv2.normalize(intensity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    inverse = cv2.GaussianBlur(255 - intensity, (3, 3), 0)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_ch = clahe.apply(intensity)
    pseudo_green = cv2.merge([
        np.clip(l_ch.astype(np.float32) * 0.65, 0, 255).astype(np.uint8),
        np.clip(l_ch.astype(np.float32) * 0.85 + inverse.astype(np.float32) * 0.45, 0, 255).astype(np.uint8),
        np.clip(l_ch.astype(np.float32) * 0.65, 0, 255).astype(np.uint8),
    ])
    print("  Reconstrução anti-LED magenta aplicada para inferência")
    return pseudo_green


def _naturalize_magenta_for_display(
    img_bgr: np.ndarray,
    _enhanced_bgr: np.ndarray,
    germ_boxes: list[tuple[int, int, int, int, float]] | None = None,
) -> np.ndarray:
    """Cria uma visualizacao naturalizada para fotos magenta extremas.

    Pinta toda massa de planta detectada pelo canal G (nao apenas dentro
    dos boxes do YOLO) sobre um substrato marrom estilizado, garantindo que
    folhas maduras ignoradas pelo detector tambem aparecam verdes ao usuario.
    """
    h, w = img_bgr.shape[:2]
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    intensity = 0.35 * b + 0.15 * g + 0.50 * r
    intensity = cv2.normalize(intensity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    light = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(intensity)
    light_f = light.astype(np.float32) / 255.0

    # Base marrom para substrato, preservando textura pela luminância.
    natural = np.zeros_like(img_bgr)
    natural[:, :, 0] = np.clip(18 + light_f * 34, 0, 255).astype(np.uint8)
    natural[:, :, 1] = np.clip(22 + light_f * 45, 0, 255).astype(np.uint8)
    natural[:, :, 2] = np.clip(28 + light_f * 58, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    grid_mask = ((lab[:, :, 0] > 150) & (hsv[:, :, 2] > 230)).astype(np.uint8) * 255
    grid_mask = cv2.morphologyEx(
        grid_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
    )
    grid_mask = cv2.morphologyEx(
        grid_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )
    natural[grid_mask > 0] = (228, 228, 220)

    # Regiões muito escuras/fora da bandeja voltam para preto/cinza.
    dark_mask = (light < 28) & (grid_mask == 0)
    natural[dark_mask] = (18, 20, 20)

    # Plant mask GLOBAL (independente do YOLO): captura folhas maduras que
    # o detector ignora sob LED magenta extremo.
    plant_mask = _magenta_plant_mask(img_bgr)
    raw_plant_coverage = float(cv2.countNonZero(plant_mask)) / max(h * w, 1)
    if raw_plant_coverage < 0.08:
        plant_mask = cv2.morphologyEx(
            plant_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
    else:
        plant_mask = cv2.morphologyEx(
            plant_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        plant_mask = cv2.morphologyEx(
            plant_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        )
    plant_mask[grid_mask > 0] = 0  # grade nunca vira planta
    plant_alpha = cv2.GaussianBlur(plant_mask, (5, 5), 0).astype(np.float32) / 255.0
    plant_alpha *= 0.55  # opacidade base do verde global

    # Boost dentro dos germ_boxes confirmados (cor mais saturada onde YOLO acertou).
    if germ_boxes:
        box_mask = np.zeros((h, w), dtype=np.uint8)
        local_plant_mask = np.zeros((h, w), dtype=np.uint8)
        source_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred_gray = cv2.GaussianBlur(source_gray, (7, 7), 0)
        for x1, y1, x2, y2, _conf in germ_boxes:
            if x2 <= x1 or y2 <= y1:
                continue
            bx1, by1 = max(0, int(x1)), max(0, int(y1))
            bx2, by2 = min(w, int(x2)), min(h, int(y2))
            if bx2 <= bx1 or by2 <= by1:
                continue
            cv2.rectangle(
                box_mask,
                (bx1, by1),
                (bx2, by2),
                255,
                -1,
            )
            crop_gray = blurred_gray[by1:by2, bx1:bx2]
            crop_grid = grid_mask[by1:by2, bx1:bx2] > 0
            if crop_gray.size == 0:
                continue
            valid_pixels = crop_gray[~crop_grid]
            if valid_pixels.size < 20:
                continue
            threshold = float(np.percentile(valid_pixels, 12))
            crop_mask = (crop_gray <= threshold) & (~crop_grid)
            local_plant_mask[by1:by2, bx1:bx2][crop_mask] = 255

        local_alpha = cv2.GaussianBlur(local_plant_mask, (7, 7), 0).astype(np.float32) / 255.0
        local_alpha *= 0.72
        plant_alpha = np.maximum(plant_alpha, local_alpha)
        # Onde box do YOLO e alguma máscara de planta coincidem, alpha sobe a 0.78.
        boosted = ((plant_mask > 0) | (local_plant_mask > 0)) & (box_mask > 0)
        plant_alpha = np.where(
            boosted,
            np.maximum(plant_alpha, 0.78),
            plant_alpha,
        )

    # Aplica pintura verde modulada pela luminância (textura preservada).
    natural_f = natural.astype(np.float32)
    leaf_color = np.zeros_like(natural_f)
    leaf_color[:, :, 0] = 24 + light_f * 22   # B suave
    leaf_color[:, :, 1] = 90 + light_f * 110  # G dominante (max ~200)
    leaf_color[:, :, 2] = 28 + light_f * 42   # R baixo

    alpha = plant_alpha[:, :, None]
    natural_f = natural_f * (1.0 - alpha) + leaf_color * alpha
    natural = np.clip(natural_f, 0, 255).astype(np.uint8)

    print("  Visualizacao naturalizada anti-magenta aplicada (plant mask global)")
    return natural


def _assess_image_quality(img_bgr: np.ndarray) -> dict:
    """Classifica iluminação crítica antes de calcular taxa de germinação."""
    b, g, r = [img_bgr[:, :, i].astype(np.float32) for i in range(3)]
    max_ch = np.max(img_bgr, axis=2)

    magenta_ratio = float(((r > g * 1.35) & (b > g * 1.05) & (r > 120)).mean())
    red_cast_ratio = float(((r > g * 1.45) & (r > b * 1.10) & (r > 130)).mean())
    purple_ratio = float(((b > g * 1.25) & (r > g * 1.10) & (b > 80)).mean())
    clipped_ratio = float((max_ch > 245).mean())
    green_to_red = float(g.mean() / max(r.mean(), 1.0))
    green_to_blue = float(g.mean() / max(b.mean(), 1.0))

    if (
        magenta_ratio >= 0.35
        or red_cast_ratio >= 0.45
        or (clipped_ratio >= 0.22 and green_to_red < 0.70 and green_to_blue < 0.85)
    ):
        return {
            "level": "low",
            "issue": "led_magenta",
            "score": round(max(0.0, 1.0 - magenta_ratio - clipped_ratio), 2),
            "warning": (
                "Foto com LED magenta/estouro de cor. Localizei o que foi possível, "
                "mas a taxa fica como leitura parcial. Para melhor precisão, use luz branca "
                "ou reduza a intensidade do LED."
            ),
        }

    if purple_ratio >= 0.20 or (green_to_blue < 0.78 and clipped_ratio >= 0.04):
        return {
            "level": "medium",
            "issue": "led_purple",
            "score": 0.72,
            "warning": (
                "Iluminação roxa detectada. A análise foi corrigida por filtros, "
                "mas uma foto com luz mais neutra tende a melhorar a contagem."
            ),
        }

    return {"level": "good", "issue": None, "score": 1.0, "warning": None}


def _enhance_for_yolo(img_bgr: np.ndarray, quality: dict) -> np.ndarray:
    """Pre-processa imagem so quando o modelo nao consegue enxergar o canal verde.

    Para led_purple e luz normal: passa imagem original (o modelo treinado em
    2026-05-26 aprendeu direto nessas fotos cruas, qualquer normalizacao desvia
    da distribuicao de treino).

    Para led_magenta extremo: precisa de _reconstruct_magenta_for_yolo porque
    G fica em ~0-10 e o modelo nao consegue achar plantas. A reconstrucao
    e SO para inferencia, nao afeta a imagem anotada de saida.

    Toggles:
    - GERMINAVISION_FORCE_NORMALIZE=1 reativa Gray World + CLAHE para luz normal/roxa
    - GERMINAVISION_RAW_MAGENTA=1 desativa a reconstrucao em magenta extremo
    """
    if os.environ.get("GERMINAVISION_FORCE_NORMALIZE") == "1":
        if quality.get("issue") == "led_magenta":
            return _reconstruct_magenta_for_yolo(img_bgr)
        return _normalize_lighting(img_bgr)

    if quality.get("issue") == "led_magenta" and os.environ.get("GERMINAVISION_RAW_MAGENTA") != "1":
        return _reconstruct_magenta_for_yolo(img_bgr)

    return img_bgr


def _plant_mask_from_hsv(hsv: np.ndarray, include_led_shadow: bool = True) -> np.ndarray:
    """Mascara de tecido vegetal em luz normal e LED roxo (NAO magenta extremo).

    Para magenta extremo, use _magenta_plant_mask que usa canal G do BGR
    diretamente (S nao distingue planta de substrato sob LED magenta puro).
    """
    bright_green = cv2.inRange(hsv, (25, 35, 45), (95, 255, 255))
    if not include_led_shadow:
        return bright_green

    # Sob LED roxo, folhas escuras migram para H 90-135 e perdem o verde classico.
    # O teto de V evita que plastico branco/rosa estourado vire planta.
    led_shadow = cv2.inRange(hsv, (80, 18, 25), (135, 255, 225))
    return cv2.bitwise_or(bright_green, led_shadow)


def _magenta_plant_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Mascara de tecido vegetal especifica para LED magenta extremo.

    Sob LED magenta puro:
    - Folhas verdes refletem PARCIALMENTE o LED e mantem residuo no canal G
      (G ~30-40) porque sao verdes em natureza.
    - Substrato seco/umido reflete o magenta intensamente (G ~5-25).
    - Grade plastica reflete tudo (G ~25-30 mas V baixo nas linhas escuras
      e V alto nas linhas claras).
    - Bandeja vazia roxa nao tem material vegetal (G ~0-10).

    O canal G do BGR diretamente separa planta de substrato; S e H sao
    igualmente altos em todas as regioes magenta.

    Threshold validado empiricamente em wa_ff81db007e67.jpeg:
    plantas G=35, substrato G=19-26, grade G=26, bandeja vazia G=5.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0]
    v_ch = hsv[:, :, 2]
    g_ch = img_bgr[:, :, 1]
    mask = (
        (h_ch >= 125) & (h_ch <= 175)
        & (g_ch >= 28)
        & (v_ch >= 110)
        & (v_ch <= 245)
    ).astype(np.uint8) * 255
    return mask


def _plant_mask_auto(
    img_bgr: np.ndarray,
    quality: dict | None = None,
) -> np.ndarray:
    """Escolhe o detector de planta correto baseado na qualidade da luz.

    Magenta extremo usa _magenta_plant_mask (canal G).
    Outros casos usam _plant_mask_from_hsv (HSV ranges classicos).
    """
    if quality is None:
        quality = _assess_image_quality(img_bgr)
    if quality.get("issue") == "led_magenta":
        return _magenta_plant_mask(img_bgr)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return _plant_mask_from_hsv(hsv, include_led_shadow=True)


def _plant_signal_ratio(hsv_crop: np.ndarray) -> float:
    if hsv_crop.size == 0:
        return 0.0
    area = hsv_crop.shape[0] * hsv_crop.shape[1]
    # Métrica conservadora usada para validar boxes: mantém o contrato antigo,
    # sem o range magenta extremo que é amplo demais para crops isolados.
    bright_green = cv2.inRange(hsv_crop, (25, 35, 45), (95, 255, 255))
    led_shadow = cv2.inRange(hsv_crop, (80, 18, 25), (135, 255, 225))
    mask = cv2.bitwise_or(bright_green, led_shadow)
    return float(cv2.countNonZero(mask)) / max(area, 1)


def _is_bright_nonplant_artifact(hsv_crop: np.ndarray) -> bool:
    """Remove reflexos/LED/plástico claro que entram na faixa roxa, mas não têm verde real."""
    if hsv_crop.size == 0:
        return False
    area = hsv_crop.shape[0] * hsv_crop.shape[1]
    v = hsv_crop[:, :, 2]
    bright_green = _plant_mask_from_hsv(hsv_crop, include_led_shadow=False)
    bright_green_ratio = float(cv2.countNonZero(bright_green)) / max(area, 1)
    clipped_ratio = float((v > 235).mean())
    return bright_green_ratio < 0.03 and (float(v.mean()) > 210 or clipped_ratio > 0.35)


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
    img_bgr: np.ndarray,
    boxes: list[tuple[str, float, tuple[int, int, int, int]]],
    w: int,
    h: int,
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """Promove folha grande sem planta associada para germinação provável."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    germ_boxes = [bbox for cls_name, _, bbox in boxes if cls_name in GERMINATION_CLASSES]
    additions: list[tuple[str, float, tuple[int, int, int, int]]] = []
    regular_min_area = max(12000, int(w * h * 0.008))
    small_led_min_area = max(3000, int(w * h * 0.002))

    for cls_name, conf, bbox in boxes:
        if cls_name != LEAF_CLASS:
            continue
        x1, y1, x2, y2 = bbox
        box_w, box_h = x2 - x1, y2 - y1
        area = _bbox_area(bbox)
        edge_margin_x = max(3, int(w * 0.01))
        edge_margin_y = max(3, int(h * 0.01))
        touches_image_edge = (
            x1 <= edge_margin_x
            or y1 <= edge_margin_y
            or x2 >= w - edge_margin_x
            or y2 >= h - edge_margin_y
        )
        if touches_image_edge and conf < 0.70:
            continue

        crop = hsv[y1:y2, x1:x2]
        if _is_bright_nonplant_artifact(crop):
            continue
        plant_ratio = _plant_signal_ratio(crop)
        regular_leaf = (
            area >= regular_min_area
            and box_w >= 70
            and box_h >= 70
            and plant_ratio >= 0.04
        )
        small_led_leaf = (
            conf >= 0.50
            and area >= small_led_min_area
            and box_w >= 45
            and box_h >= 45
            and plant_ratio >= 0.12
        )
        if not (regular_leaf or small_led_leaf):
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
    quality: dict | None = None,
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """Promove agrupamentos verdes grandes que o YOLO viu como planta/folha parcial."""
    h, w = img_bgr.shape[:2]
    if quality is None:
        quality = _assess_image_quality(img_bgr)
    if quality.get("issue") == "led_magenta":
        mask = _plant_mask_auto(img_bgr, quality)
    else:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = _plant_mask_from_hsv(hsv, include_led_shadow=False)
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
    img_h, img_w = img_bgr.shape[:2]
    kept: list[tuple[str, float, tuple[int, int, int, int]]] = []
    removed = 0

    for item in boxes:
        cls_name, conf, bbox = item
        if cls_name not in GERMINATION_CLASSES:
            kept.append(item)
            continue

        x1, y1, x2, y2 = bbox
        area = _bbox_area(bbox)
        crop = hsv[y1:y2, x1:x2]
        if crop.size == 0 or area <= 0:
            removed += 1
            continue
        bright_mask = _plant_mask_from_hsv(crop, include_led_shadow=False)
        bright_ratio = cv2.countNonZero(bright_mask) / max(area, 1)
        plant_ratio = _plant_signal_ratio(crop)
        if plant_ratio < 0.04 or _is_bright_nonplant_artifact(crop):
            removed += 1
            continue
        if bright_ratio < 0.005 and area > 30000:
            removed += 1
            continue
        if bright_ratio < 0.03 and plant_ratio < 0.20:
            removed += 1
            continue
        touches_edge = x1 <= 2 or y1 <= 2 or x2 >= img_w - 2 or y2 >= img_h - 2
        if bright_ratio < 0.03 and (area < 2500 or (touches_edge and area < 5000)):
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
        mask = _plant_mask_from_hsv(crop, include_led_shadow=False)
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


def _projection_bands(
    projection: np.ndarray,
    length: int,
    min_width: int = 3,
) -> list[tuple[int, int, float, float]]:
    """Agrupa picos de uma projeção 1D em faixas candidatas de linha da grade."""
    if projection.size == 0:
        return []

    smooth_k = max(5, int(length * 0.015) | 1)
    smooth = np.convolve(projection, np.ones(smooth_k) / smooth_k, mode="same")
    peak = float(smooth.max())
    if peak <= 0:
        return []

    threshold = max(0.05, min(0.24, peak * 0.35))
    above = smooth > threshold
    bands: list[tuple[int, int, float, float]] = []
    start: int | None = None

    for idx, is_above in enumerate(above):
        if is_above and start is None:
            start = idx
        at_end = idx == len(above) - 1
        if (not is_above or at_end) and start is not None:
            end = idx if not is_above else idx + 1
            if end - start >= min_width:
                center = (start + end - 1) / 2.0
                bands.append((start, end, center, float(smooth[start:end].max())))
            start = None

    return bands


def _merge_nearby_bands(
    bands: list[tuple[int, int, float, float]],
    axis_length: int,
) -> list[tuple[int, int, float, float]]:
    """Une faixas muito próximas produzidas por sujeira/reflexo na mesma linha."""
    merged: list[tuple[int, int, float, float]] = []
    for band in sorted(bands, key=lambda item: item[2]):
        if band[2] < axis_length * 0.02 or band[2] > axis_length * 0.98:
            continue
        if merged and band[2] - merged[-1][2] < axis_length * 0.055:
            prev = merged[-1]
            merged[-1] = (
                min(prev[0], band[0]),
                max(prev[1], band[1]),
                (prev[2] + band[2]) / 2.0,
                max(prev[3], band[3]),
            )
        else:
            merged.append(band)
    return merged


def _regular_grid_subset(
    bands: list[tuple[int, int, float, float]],
    axis_length: int,
) -> list[tuple[int, int, float, float]]:
    """Seleciona a maior sequência com espaçamento regular, descartando bordas falsas."""
    if len(bands) <= 2:
        return bands

    min_gap = axis_length * 0.08
    best: tuple[int, float, float, int, int] | None = None

    for start in range(len(bands)):
        for end in range(start + 2, len(bands) + 1):
            subset = bands[start:end]
            centers = np.array([band[2] for band in subset], dtype=np.float32)
            gaps = np.diff(centers)
            if gaps.size == 0:
                continue
            median_gap = float(np.median(gaps))
            if median_gap < min_gap:
                continue
            deviation = float(np.max(np.abs(gaps - median_gap)) / max(median_gap, 1.0))
            if deviation > 0.32:
                continue
            # Prioriza mais linhas; em empate, prefere o espaçamento maior/mais regular.
            score = (len(subset), median_gap, -deviation, start, end)
            if best is None or score[:3] > best[:3]:
                best = score

    if best is None:
        return []
    return bands[best[3]:best[4]]


def _edge_interval_has_grid_signal(
    bright_mask: np.ndarray,
    axis: str,
    start: int,
    end: int,
    perpendicular_bands: list[tuple[int, int, float, float]],
) -> bool:
    """Confirma se uma faixa de borda ainda contém células, não apenas fundo/vaso."""
    if end <= start or len(perpendicular_bands) < 2:
        return False

    h, w = bright_mask.shape[:2]
    hits = 0
    required = max(2, int(np.ceil(len(perpendicular_bands) * 0.35)))

    for band in perpendicular_bands:
        center = int(round(band[2]))
        if axis == "x":
            x1, x2 = max(0, start), min(w, end)
            y1, y2 = max(0, center - 8), min(h, center + 9)
        else:
            x1, x2 = max(0, center - 8), min(w, center + 9)
            y1, y2 = max(0, start), min(h, end)

        roi = bright_mask[y1:y2, x1:x2]
        if roi.size and float(roi.mean()) / 255.0 > 0.08:
            hits += 1

    return hits >= required


def _count_axis_slots(
    bands: list[tuple[int, int, float, float]],
    axis_length: int,
    bright_mask: np.ndarray,
    axis: str,
    perpendicular_bands: list[tuple[int, int, float, float]],
) -> int | None:
    """Conta intervalos de células ao longo de um eixo a partir das linhas da grade."""
    intervals = _axis_cell_intervals(bands, axis_length, bright_mask, axis, perpendicular_bands)
    return len(intervals) if intervals else None


def _axis_cell_intervals(
    bands: list[tuple[int, int, float, float]],
    axis_length: int,
    bright_mask: np.ndarray,
    axis: str,
    perpendicular_bands: list[tuple[int, int, float, float]],
) -> list[tuple[int, int]]:
    """Retorna intervalos de células válidas ao longo de um eixo da grade."""
    if len(bands) < 2:
        return []

    centers = [band[2] for band in bands]
    spacing = float(np.median(np.diff(np.array(centers, dtype=np.float32))))
    if spacing <= 0:
        return []

    # Bordas só contam quando parecem quase uma célula inteira; margens largas
    # ou lascas estreitas de bandeja cortada entram como recorte, não célula útil.
    edge_min = max(8, int(spacing * 0.50))
    edge_max = int(spacing * 1.15)
    intervals: list[tuple[int, int]] = []

    leading_end = int(round(centers[0]))
    if edge_min <= leading_end <= edge_max and _edge_interval_has_grid_signal(
        bright_mask, axis, 0, leading_end, perpendicular_bands
    ):
        intervals.append((0, leading_end))

    for start, end in zip(centers, centers[1:]):
        start_i, end_i = int(round(start)), int(round(end))
        if end_i > start_i:
            intervals.append((start_i, end_i))

    trailing_start = int(round(centers[-1]))
    trailing_len = axis_length - trailing_start
    if edge_min <= trailing_len <= edge_max and _edge_interval_has_grid_signal(
        bright_mask, axis, trailing_start, axis_length, perpendicular_bands
    ):
        intervals.append((trailing_start, axis_length))

    return intervals


def _grid_bands_from_image(
    img_bgr: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int, float, float]], list[tuple[int, int, float, float]]]:
    """Extrai máscara clara e linhas regulares da grade."""
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    bright_grid = ((value > 125) & (saturation < 135) & (lightness > 115)).astype(np.uint8) * 255
    green = cv2.inRange(hsv, (25, 35, 35), (95, 255, 255))
    bright_grid = cv2.bitwise_and(bright_grid, cv2.bitwise_not(green))
    bright_grid = cv2.morphologyEx(
        bright_grid,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(30, int(h * 0.08))))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, int(w * 0.08)), 3))
    vertical_mask = cv2.morphologyEx(bright_grid, cv2.MORPH_OPEN, vertical_kernel)
    horizontal_mask = cv2.morphologyEx(bright_grid, cv2.MORPH_OPEN, horizontal_kernel)

    vertical_bands = _regular_grid_subset(
        _merge_nearby_bands(_projection_bands(vertical_mask.mean(axis=0) / 255.0, w), w),
        w,
    )
    horizontal_bands = _regular_grid_subset(
        _merge_nearby_bands(_projection_bands(horizontal_mask.mean(axis=1) / 255.0, h), h),
        h,
    )

    return bright_grid, vertical_bands, horizontal_bands


def _detect_grid_via_edges(
    img_bgr: np.ndarray,
    quality: dict | None = None,
) -> tuple[list[int], list[int]] | None:
    """Detecta posicoes da grade via edges + Hough lines.

    Retorna (vertical_xs, horizontal_ys) ou None se nao houver grade
    detectavel. Funciona em luz normal, roxa e magenta porque depende de
    contraste local (gradient), nao threshold absoluto de cor/brilho.
    """
    h, w = img_bgr.shape[:2]

    # Equalizacao local para realcar o gradient da grade vs interior da cell.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(gray)

    # Thresholds baixos pegam edges sutis sob LED magenta.
    edges = cv2.Canny(enhanced, 15, 60, apertureSize=3)

    min_line_len = max(25, int(min(h, w) * 0.05))
    max_line_gap = max(12, int(min(h, w) * 0.025))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=25,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )
    if lines is None or len(lines) < 4:
        return None

    raw_vertical = 0
    raw_horizontal = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dx == 0 and dy == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        if angle <= 15:
            raw_horizontal += 1
        elif angle >= 75:
            raw_vertical += 1

    if raw_vertical < 3 or raw_horizontal < 3:
        return None

    grad_x = np.abs(cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3))
    grad_y = np.abs(cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3))
    y_cut = max(1, int(h * 0.84))
    score_x = grad_x[:y_cut, :].mean(axis=0) + edges[:y_cut, :].mean(axis=0) * 0.6
    score_y = grad_y.mean(axis=1) + edges.mean(axis=1) * 0.6

    def normalize_score(score: np.ndarray) -> np.ndarray:
        smooth = np.convolve(score.astype(float), np.ones(7) / 7, mode="same")
        low = float(np.percentile(smooth, 30))
        high = float(np.percentile(smooth, 98))
        return np.clip((smooth - low) / max(high - low, 1e-6), 0.0, 1.0)

    min_dim = min(h, w)
    min_spacing = max(60, int(min_dim * 0.075))
    max_spacing = min(170, max(125, int(min_dim * 0.18)))

    def peak_positions(norm_score: np.ndarray) -> list[tuple[int, float]]:
        min_dist = max(35, int(min_spacing * 0.55))
        threshold = max(float(np.percentile(norm_score, 72)), float(norm_score.max()) * 0.45)
        remaining = norm_score.copy()
        peaks: list[tuple[int, float]] = []
        for _ in range(40):
            idx = int(np.argmax(remaining))
            value = float(remaining[idx])
            if value < threshold:
                break
            peaks.append((idx, value))
            lo = max(0, idx - min_dist)
            hi = min(len(remaining), idx + min_dist + 1)
            remaining[lo:hi] = 0
        return sorted(peaks)

    def spacing_candidates(peaks: list[tuple[int, float]]) -> list[tuple[float, int]]:
        hist = np.zeros(max_spacing + 1, dtype=float)
        for i, (a, va) in enumerate(peaks):
            for b, vb in peaks[i + 1:]:
                distance = b - a
                for divisor in (1, 2, 3):
                    spacing = distance / divisor
                    if min_spacing <= spacing <= max_spacing:
                        center = int(round(spacing))
                        weight = math.sqrt(va * vb) / (divisor ** 0.25)
                        lo = max(min_spacing, center - 3)
                        hi = min(max_spacing, center + 3)
                        for s in range(lo, hi + 1):
                            hist[s] += weight * (1.0 - abs(s - spacing) / 4.0)

        smooth = np.convolve(hist, np.ones(5) / 5, mode="same")
        ranked = sorted(
            ((float(smooth[s]), s) for s in range(min_spacing, max_spacing + 1) if smooth[s] > 0),
            reverse=True,
        )
        selected: list[tuple[float, int]] = []
        for vote, spacing in ranked:
            if all(abs(spacing - chosen) > 8 for _, chosen in selected):
                selected.append((vote, spacing))
            if len(selected) >= 8:
                break
        return selected

    def choose_spacing_pair(
        x_candidates: list[tuple[float, int]],
        y_candidates: list[tuple[float, int]],
    ) -> tuple[int, int] | None:
        best: tuple[float, int, int] | None = None
        for vote_x, spacing_x in x_candidates:
            for vote_y, spacing_y in y_candidates:
                ratio = spacing_x / max(spacing_y, 1)
                if not (0.65 <= ratio <= 1.45):
                    continue
                score = vote_x + vote_y - 12.0 * abs(math.log(ratio))
                item = (score, spacing_x, spacing_y)
                if best is None or item > best:
                    best = item
        if best is None:
            return None
        return best[1], best[2]

    def fit_axis_positions(
        norm_score: np.ndarray,
        peaks: list[tuple[int, float]],
        axis_len: int,
        spacing: int,
        min_lines: int,
    ) -> list[int]:
        peak_values = {position: value for position, value in peaks}
        peak_pos = [position for position, _ in peaks]
        best: tuple[float, list[int]] | None = None
        start_spacing = max(45, int(spacing * 0.90))
        end_spacing = int(spacing * 1.10)

        for current_spacing in range(start_spacing, end_spacing + 1):
            tolerance = max(10, int(current_spacing * 0.30))
            for offset in range(0, current_spacing, 2):
                hits: list[tuple[int, int, float, float]] = []
                k = 0
                predicted = offset
                while predicted < axis_len:
                    if axis_len * 0.01 <= predicted <= axis_len * 0.99:
                        close = [p for p in peak_pos if abs(p - predicted) <= tolerance]
                        if close:
                            peak = max(close, key=lambda p: peak_values[p])
                            hits.append(
                                (
                                    k,
                                    peak,
                                    peak_values[peak],
                                    abs(float(peak) - float(predicted)),
                                )
                            )
                    k += 1
                    predicted = offset + k * current_spacing

                if len(hits) < min_lines:
                    continue
                first, last = hits[0][0], hits[-1][0]
                centers = [
                    int(round(offset + k * current_spacing))
                    for k in range(first, last + 1)
                    if 0 <= offset + k * current_spacing < axis_len
                ]
                if len(centers) < min_lines:
                    continue
                coverage = len({hit[0] for hit in hits}) / len(centers)
                if coverage < 0.52:
                    continue
                median_signal = float(np.median([hit[2] for hit in hits]))
                residual = float(np.median([hit[3] for hit in hits])) / max(current_spacing, 1)
                axis_score = (
                    len(centers) * 0.55
                    + len(hits) * 1.15
                    + coverage * 2.0
                    + median_signal * 2.0
                    - residual * 5.0
                )
                item = (axis_score, centers)
                if best is None or item[0] > best[0]:
                    best = item

        return best[1] if best is not None else []

    norm_x = normalize_score(score_x)
    norm_y = normalize_score(score_y)
    peaks_x = peak_positions(norm_x)
    peaks_y = peak_positions(norm_y)
    spacing_pair = choose_spacing_pair(
        spacing_candidates(peaks_x),
        spacing_candidates(peaks_y),
    )
    if spacing_pair is None:
        return None

    spacing_x, spacing_y = spacing_pair
    vert_centers = fit_axis_positions(norm_x, peaks_x, w, spacing_x, min_lines=6)
    horiz_centers = fit_axis_positions(norm_y, peaks_y, h, spacing_y, min_lines=8)

    if len(vert_centers) < 6 or len(horiz_centers) < 8:
        return None

    # Regulariza pelo stride mediano: força grade fisica uniforme,
    # preenche gaps de Hough e evita linhas locais off-center.
    vert_centers = _regularize_grid_lines(vert_centers, w)
    horiz_centers = _regularize_grid_lines(horiz_centers, h)

    if len(vert_centers) < 3 or len(horiz_centers) < 3:
        return None

    return vert_centers, horiz_centers


def _regularize_grid_lines(positions: list[int], axis_length: int) -> list[int]:
    """Gera grade regular usando stride mediano global."""
    if len(positions) < 3:
        return positions

    sorted_pos = sorted(positions)
    diffs = np.diff(sorted_pos)
    if len(diffs) == 0:
        return positions
    stride = float(np.median(diffs))
    if stride < 10:
        return positions

    # Ancora pragmaticamente na mediana das linhas detectadas.
    anchor = float(np.median(sorted_pos))

    regularized: list[float] = [anchor]
    p = anchor - stride
    while p > -stride * 0.4:
        regularized.insert(0, p)
        p -= stride
    p = anchor + stride
    while p < axis_length + stride * 0.4:
        regularized.append(p)
        p += stride

    valid = [int(round(x)) for x in regularized if -2 <= x <= axis_length + 2]
    valid = [max(0, min(axis_length, x)) for x in valid]
    valid = sorted(set(valid))
    if len(valid) < 3:
        return positions

    snap_tol = stride * 0.3
    snapped: list[int] = []
    for regular in valid:
        candidates = [detected for detected in sorted_pos if abs(detected - regular) < snap_tol]
        if candidates:
            snapped.append(min(candidates, key=lambda detected: abs(detected - regular)))
        else:
            snapped.append(regular)
    return sorted(set(snapped))


def _grid_cell_boxes_from_edges(
    img_bgr: np.ndarray,
    quality: dict | None = None,
) -> list[tuple[int, int, int, int]]:
    """Gera bboxes de cells a partir das linhas detectadas via edges."""
    detected = _detect_grid_via_edges(img_bgr, quality)
    if detected is None:
        return []
    vert_xs, horiz_ys = detected
    cells: list[tuple[int, int, int, int]] = []
    for i in range(len(horiz_ys) - 1):
        for j in range(len(vert_xs) - 1):
            x1, x2 = vert_xs[j], vert_xs[j + 1]
            y1, y2 = horiz_ys[i], horiz_ys[i + 1]
            if x2 > x1 and y2 > y1:
                cells.append((x1, y1, x2, y2))
    return cells


def _grid_cell_boxes_magenta(img_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Estima cells dividindo a ROI da bandeja magenta em grade uniforme.

    Usado como fallback quando _grid_cell_boxes, que depende de bandas
    brilhantes, falha em fotos com LED magenta muito saturado.

    Estima número de colunas/linhas pelo lado curto/longo da ROI assumindo
    células aproximadamente quadradas, e pelo metadado de bandeja padrão
    (TRAY_CAPACITY=200 sugere ~7-8 cols x ~25-28 rows mas para fotos
    parciais frequentemente vemos 5-7 cols x 6-9 rows visíveis).
    """
    roi = _magenta_grid_roi(img_bgr)
    if roi is None:
        return []
    rx1, ry1, rx2, ry2 = roi
    roi_w, roi_h = rx2 - rx1, ry2 - ry1
    if roi_w <= 30 or roi_h <= 30:
        return []

    # Aproxima células quadradas. Range plausível: 5-9 cols visíveis.
    cell_size_est = int(min(roi_w, roi_h) / 6)
    cell_size_est = max(60, min(180, cell_size_est))

    cols = max(2, int(round(roi_w / cell_size_est)))
    rows = max(2, int(round(roi_h / cell_size_est)))
    cell_w = roi_w / cols
    cell_h = roi_h / rows

    cells: list[tuple[int, int, int, int]] = []
    for row in range(rows):
        for col in range(cols):
            x1 = int(round(rx1 + col * cell_w))
            y1 = int(round(ry1 + row * cell_h))
            x2 = int(round(rx1 + (col + 1) * cell_w))
            y2 = int(round(ry1 + (row + 1) * cell_h))
            if x2 > x1 and y2 > y1:
                cells.append((x1, y1, x2, y2))
    return cells


def _grid_cell_boxes(
    img_bgr: np.ndarray,
    quality: dict | None = None,
) -> list[tuple[int, int, int, int]]:
    """Retorna boxes aproximadas das celulas da grade.

    Estrategia:
    1. Bands brilhantes para luz normal/roxa com grade clara.
    2. Edges + Hough para magenta extremo, onde cor/brilho absoluto falha.
    """
    h, w = img_bgr.shape[:2]
    bright_grid, vertical_bands, horizontal_bands = _grid_bands_from_image(img_bgr)
    x_intervals = _axis_cell_intervals(vertical_bands, w, bright_grid, "x", horizontal_bands)
    y_intervals = _axis_cell_intervals(horizontal_bands, h, bright_grid, "y", vertical_bands)
    if x_intervals and y_intervals:
        cells_bands = [
            (x1, y1, x2, y2)
            for y1, y2 in y_intervals
            for x1, x2 in x_intervals
            if x2 > x1 and y2 > y1
        ]
        if quality is None or quality.get("issue") != "led_magenta" or len(cells_bands) >= 20:
            return cells_bands
    else:
        cells_bands = []

    cells_edges = _grid_cell_boxes_from_edges(img_bgr, quality)
    if cells_edges:
        return cells_edges

    return cells_bands


def _tiny_green_germination_fallback(
    img_bgr: np.ndarray,
    boxes: list[tuple[str, float, tuple[int, int, int, int]]],
    quality: dict | None = None,
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """Adiciona mudas muito pequenas quando a célula da grade está vazia."""
    h, w = img_bgr.shape[:2]
    if quality is None:
        quality = _assess_image_quality(img_bgr)
    cells = _grid_cell_boxes(img_bgr, quality)
    if not cells:
        cells = _grid_cell_boxes_magenta(img_bgr)
        if not cells:
            return boxes

    if quality.get("issue") == "led_magenta":
        mask = _plant_mask_auto(img_bgr, quality)
    else:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = _plant_mask_from_hsv(hsv, include_led_shadow=False)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    close_k = max(9, int(min(h, w) * 0.028) | 1)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)),
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    germ_boxes = [bbox for cls_name, _, bbox in boxes if cls_name in GERMINATION_CLASSES]
    additions: list[tuple[str, float, tuple[int, int, int, int]]] = []
    img_area = float(h * w)
    min_area = max(55.0, img_area * 0.000035)
    max_area = max(1200.0, img_area * 0.0025)
    max_dim = max(45, int(min(h, w) * 0.10))

    def _cell_for_center(cx: float, cy: float) -> tuple[int, int, int, int] | None:
        for cell in cells:
            x1, y1, x2, y2 = cell
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return cell
        return None

    def _box_center_inside(box: tuple[int, int, int, int], cell: tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return cell[0] <= cx <= cell[2] and cell[1] <= cy <= cell[3]

    for contour in contours:
        area = cv2.contourArea(contour)
        if not (min_area <= area <= max_area):
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 8 or bh < 8 or bw > max_dim or bh > max_dim:
            continue
        fill_ratio = area / max(bw * bh, 1)
        if fill_ratio < 0.12:
            continue

        cx, cy = x + bw / 2.0, y + bh / 2.0
        cell = _cell_for_center(cx, cy)
        if cell is None:
            continue
        if any(_box_center_inside(germ_bbox, cell) for germ_bbox in germ_boxes):
            continue

        crop_mask = mask[y:y + bh, x:x + bw]
        bright_ratio = cv2.countNonZero(crop_mask) / max(bw * bh, 1)
        if bright_ratio < 0.08:
            continue

        bbox = _expand_bbox((x, y, x + bw, y + bh), w, h, ratio=0.35)
        if any(_is_duplicate_germination(bbox, germ_bbox) for germ_bbox in germ_boxes):
            continue
        if any(_is_duplicate_germination(bbox, item[2]) for item in additions):
            continue
        additions.append(("Germinacao", 0.35, bbox))

    if additions:
        print(f"  [germ] {len(additions)} mini muda(s) adicionada(s) por grade+verde")
    return boxes + additions


def _grid_occupation_germination_fallback(
    img_bgr: np.ndarray,
    boxes: list[tuple[str, float, tuple[int, int, int, int]]],
    quality: dict | None = None,
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """Promove celulas da grade com massa verde acima do limiar.

    Cobre folhas maduras sobrepostas que YOLO ignora e tiny_green descarta
    por area maxima. Usa a grade detectada como ancora.
    """
    h, w = img_bgr.shape[:2]
    if quality is None:
        quality = _assess_image_quality(img_bgr)
    cells = _grid_cell_boxes(img_bgr, quality)
    if not cells:
        cells = _grid_cell_boxes_magenta(img_bgr)
        if not cells:
            return boxes

    plant_mask = _plant_mask_auto(img_bgr, quality)
    raw_mask_coverage = float(cv2.countNonZero(plant_mask)) / max(h * w, 1)
    sparse_magenta = quality.get("issue") == "led_magenta" and raw_mask_coverage < 0.08

    plant_mask = cv2.morphologyEx(
        plant_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    plant_mask = cv2.morphologyEx(
        plant_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (11, 11) if sparse_magenta else (5, 5),
        ),
    )

    germ_boxes = [bbox for cls_name, _, bbox in boxes if cls_name in GERMINATION_CLASSES]
    coverage_threshold = 0.025 if sparse_magenta else 0.18

    def _box_center_inside(
        box: tuple[int, int, int, int],
        cell: tuple[int, int, int, int],
    ) -> bool:
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return cell[0] <= cx <= cell[2] and cell[1] <= cy <= cell[3]

    additions: list[tuple[str, float, tuple[int, int, int, int]]] = []
    for cell in cells:
        cx1, cy1, cx2, cy2 = cell
        cell_w, cell_h = cx2 - cx1, cy2 - cy1
        if cell_w <= 12 or cell_h <= 12:
            continue
        margin_x = max(1, int(cell_w * 0.10))
        margin_y = max(1, int(cell_h * 0.10))
        ix1, iy1 = cx1 + margin_x, cy1 + margin_y
        ix2, iy2 = cx2 - margin_x, cy2 - margin_y
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        cell_mask = plant_mask[iy1:iy2, ix1:ix2]
        if cell_mask.size == 0:
            continue
        coverage = float(cv2.countNonZero(cell_mask)) / max(cell_mask.size, 1)
        if coverage < coverage_threshold:
            continue
        if any(_box_center_inside(germ_bbox, cell) for germ_bbox in germ_boxes):
            continue
        if any(_box_center_inside(item[2], cell) for item in additions):
            continue
        ys, xs = np.where(cell_mask > 0)
        if ys.size == 0:
            continue
        bx1 = ix1 + int(xs.min())
        by1 = iy1 + int(ys.min())
        bx2 = ix1 + int(xs.max()) + 1
        by2 = iy1 + int(ys.max()) + 1
        bbox = _expand_bbox((bx1, by1, bx2, by2), w, h, ratio=0.10)
        conf = round(min(0.65, 0.32 + max(0.0, coverage - coverage_threshold)), 3)
        additions.append(("Germinacao", conf, bbox))

    if additions:
        print(
            f"  [germ] {len(additions)} germinacao(es) por grid occupation "
            f"(coverage>={coverage_threshold:.3f})"
        )
    return boxes + additions


def _cluster_based_germination_fallback(
    img_bgr: np.ndarray,
    boxes: list[tuple[str, float, tuple[int, int, int, int]]],
    quality: dict | None = None,
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """Promove clusters contiguos grandes de plant_mask a germinacoes.

    Substitui a heuristica de "1 planta por cell ocupada" que infla a contagem
    quando folhas grandes extrapolam pra cells vizinhas. Cada cluster contiguo
    corresponde a UMA planta, independente de quantas cells visualmente cruza.
    """
    h, w = img_bgr.shape[:2]
    img_area = float(h * w)

    if quality is None:
        quality = _assess_image_quality(img_bgr)
    if quality.get("issue") == "led_magenta":
        plant_mask = _magenta_plant_mask(img_bgr)
    else:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        plant_mask = _plant_mask_from_hsv(hsv, include_led_shadow=True)
    raw_mask_coverage = float(cv2.countNonZero(plant_mask)) / max(int(img_area), 1)
    sparse_magenta = quality.get("issue") == "led_magenta" and raw_mask_coverage < 0.08
    # Morphology pra unir folhas separadas da mesma planta e suavizar borda.
    plant_mask = cv2.morphologyEx(
        plant_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    close_k = (
        5
        if sparse_magenta
        else min(5, max(3, int(min(h, w) * 0.004) | 1))
    )
    plant_mask = cv2.morphologyEx(
        plant_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)),
    )

    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        plant_mask, connectivity=8
    )

    germ_boxes = [bbox for cls_name, _, bbox in boxes if cls_name in GERMINATION_CLASSES]
    additions: list[tuple[str, float, tuple[int, int, int, int]]] = []

    min_area = (
        max(45.0, img_area * 0.00003)
        if sparse_magenta
        else max(400.0, img_area * 0.0004)
    )  # ~0.04% da imagem no caso normal.
    min_dim = 14 if sparse_magenta else 20
    max_area = img_area * 0.15  # Cluster maior que 15% provavelmente eh varios juntos.

    for label_id in range(1, n_labels):
        x, y, bw, bh, area = stats[label_id]
        if area < min_area or area > max_area:
            continue
        if bw < min_dim or bh < min_dim:
            continue
        cx, cy = centroids[label_id]

        covered = False
        for gx1, gy1, gx2, gy2 in germ_boxes:
            if gx1 <= cx <= gx2 and gy1 <= cy <= gy2:
                # YOLO ja viu, nao duplica.
                covered = True
                break
        if covered:
            continue
        # Usa o nucleo do cluster para nao colapsar plantas vizinhas no
        # dedupe final quando folhas grandes invadem a cell ao lado.
        core_w = max(20, int(round(float(bw) * 0.45)))
        core_h = max(20, int(round(float(bh) * 0.45)))
        bx1 = max(0, int(round(float(cx) - core_w / 2.0)))
        by1 = max(0, int(round(float(cy) - core_h / 2.0)))
        bx2 = min(w, int(round(float(cx) + core_w / 2.0)))
        by2 = min(h, int(round(float(cy) + core_h / 2.0)))
        if bx2 <= bx1 or by2 <= by1:
            continue
        bbox = (bx1, by1, bx2, by2)
        area_norm = min(1.0, area / (img_area * 0.01))
        conf = round(min(0.70, 0.40 + area_norm * 0.30), 3)
        additions.append(("Germinacao", conf, bbox))

    if additions:
        print(f"  [germ] {len(additions)} germinacao(es) por cluster contiguo (connected components)")
    return boxes + additions


def _hybrid_grid_cluster_fallback(
    img_bgr: np.ndarray,
    boxes: list[tuple[str, float, tuple[int, int, int, int]]],
    quality: dict | None = None,
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """Contagem hibrida: 1 planta por cluster CONTIGUO DENTRO de cada cell.

    Combina o melhor de grid_occupation (ancorar contagem na grade pra
    rejeitar regioes fora) e cluster_based (1 planta por massa continua,
    ao inves de 1 por cell ocupada). Cells vazias sao ignoradas; cells
    com 2+ clusters distintos contam multiplas plantas; folhas que
    extrapolam pra cell vizinha NAO inflam (a porcao fora da cell eh
    cortada pelo bounding da cell).
    """
    h, w = img_bgr.shape[:2]
    img_area = float(h * w)

    if quality is None:
        quality = _assess_image_quality(img_bgr)

    cells = _grid_cell_boxes(img_bgr, quality)
    magenta_cells: list[tuple[int, int, int, int]] = []
    if quality.get("issue") == "led_magenta" or not cells:
        magenta_cells = _grid_cell_boxes_magenta(img_bgr)
    if magenta_cells and (
        not cells
        or (
            quality.get("issue") == "led_magenta"
            and (len(cells) < 8 or len(cells) > len(magenta_cells) * 1.25)
        )
    ):
        cells = magenta_cells
    if not cells:
        return boxes

    plant_mask = _plant_mask_auto(img_bgr, quality)
    is_magenta = quality.get("issue") == "led_magenta"
    plant_mask = cv2.morphologyEx(
        plant_mask,
        cv2.MORPH_CLOSE if is_magenta else cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    germ_boxes = [bbox for cls_name, _, bbox in boxes if cls_name in GERMINATION_CLASSES]

    def _box_center_inside(
        box: tuple[int, int, int, int],
        cell: tuple[int, int, int, int],
    ) -> bool:
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return cell[0] <= cx <= cell[2] and cell[1] <= cy <= cell[3]

    additions: list[tuple[str, float, tuple[int, int, int, int]]] = []

    # Min area pra cluster local dentro de uma cell. Escala com tamanho de
    # cell tipico nas fotos do projeto.
    if cells:
        cell_areas = [(cell[2] - cell[0]) * (cell[3] - cell[1]) for cell in cells]
        median_cell_area = float(sorted(cell_areas)[len(cell_areas) // 2])
    else:
        median_cell_area = img_area / 50
    min_cluster_area = max(80.0, median_cell_area * (0.02 if is_magenta else 0.04))
    coverage_threshold = 0.05 if is_magenta else 0.10

    for cell in cells:
        cx1, cy1, cx2, cy2 = cell
        cell_w, cell_h = cx2 - cx1, cy2 - cy1
        if cell_w <= 10 or cell_h <= 10:
            continue

        # Mascara local da cell com pequena margem interna pra evitar
        # borda da grade contaminar.
        mx = max(1, int(cell_w * 0.06))
        my = max(1, int(cell_h * 0.06))
        ix1, iy1 = cx1 + mx, cy1 + my
        ix2, iy2 = cx2 - mx, cy2 - my
        if ix2 <= ix1 or iy2 <= iy1:
            continue

        cell_mask = plant_mask[iy1:iy2, ix1:ix2]
        if cell_mask.size == 0:
            continue
        coverage = float(cv2.countNonZero(cell_mask)) / max(cell_mask.size, 1)
        if coverage < coverage_threshold:
            continue

        # Connected components DENTRO da cell: cada cluster eh uma planta.
        n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            cell_mask,
            connectivity=8,
        )

        clusters_added_in_cell = 0
        for label_id in range(1, n_labels):
            cx, cy, cw, ch, area = stats[label_id]
            if area < min_cluster_area:
                continue
            min_cluster_dim = 4 if is_magenta else 8
            if cw < min_cluster_dim or ch < min_cluster_dim:
                continue

            abs_x1 = ix1 + int(cx)
            abs_y1 = iy1 + int(cy)
            abs_x2 = abs_x1 + int(cw)
            abs_y2 = abs_y1 + int(ch)
            bbox = _expand_bbox((abs_x1, abs_y1, abs_x2, abs_y2), w, h, ratio=0.05)

            # Skipa se ja tem germ_box do YOLO cobrindo.
            covered = any(
                gx1 <= (abs_x1 + abs_x2) / 2.0 <= gx2
                and gy1 <= (abs_y1 + abs_y2) / 2.0 <= gy2
                for gx1, gy1, gx2, gy2 in germ_boxes
            )
            if covered:
                continue

            area_norm = min(1.0, area / max(median_cell_area * 0.5, 1.0))
            conf = round(min(0.65, 0.38 + area_norm * 0.25), 3)
            additions.append(("Germinacao", conf, bbox))
            clusters_added_in_cell += 1

        # Safety removida em 2026-05-26: a heuristica de usar a massa verde
        # inteira da cell como bbox gerava bboxes "celulares" cobrindo varias
        # plantas separadas. O modelo novo (treinado 2026-05-26, mAP50 0.85)
        # confia em si proprio para esses casos. Pra reativar (rollback),
        # exporte GERMINAVISION_HYBRID_CELL_SAFETY=1.
        if (
            clusters_added_in_cell == 0
            and coverage >= coverage_threshold
            and os.environ.get("GERMINAVISION_HYBRID_CELL_SAFETY") == "1"
        ):
            ys, xs = np.where(cell_mask > 0)
            if ys.size > 0:
                bx1 = ix1 + int(xs.min())
                by1 = iy1 + int(ys.min())
                bx2 = ix1 + int(xs.max()) + 1
                by2 = iy1 + int(ys.max()) + 1
                bbox = _expand_bbox((bx1, by1, bx2, by2), w, h, ratio=0.05)
                conf = round(min(0.55, 0.30 + coverage * 0.50), 3)
                additions.append(("Germinacao", conf, bbox))

    if additions:
        print(f"  [germ] {len(additions)} planta(s) por hybrid grid+cluster (1 por cluster intra-cell)")
    return boxes + additions


def _magenta_grid_roi(img_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Estima a área útil da bandeja em fotos magenta extremas."""
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
    mask = (lightness > 140).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    vertical_mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(20, int(h * 0.05)))),
    )
    horizontal_mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(w * 0.05)), 3)),
    )
    vertical_bands = _merge_nearby_bands(_projection_bands(vertical_mask.mean(axis=0) / 255.0, w, min_width=2), w)
    horizontal_bands = _merge_nearby_bands(_projection_bands(horizontal_mask.mean(axis=1) / 255.0, h, min_width=2), h)
    if len(vertical_bands) < 3 or len(horizontal_bands) < 3:
        return None

    def _bounds(bands: list[tuple[int, int, float, float]], length: int) -> tuple[int, int]:
        centers = np.array([band[2] for band in bands], dtype=np.float32)
        gaps = np.diff(centers)
        spacing = float(np.median(gaps[gaps < length * 0.18])) if np.any(gaps < length * 0.18) else float(np.median(gaps))
        margin = max(12, int(spacing * 0.55))
        return max(0, int(round(float(centers[0]) - margin))), min(length, int(round(float(centers[-1]) + margin)))

    x1, x2 = _bounds(vertical_bands, w)
    y1, y2 = _bounds(horizontal_bands, h)
    return (x1, y1, x2, y2)


def _filter_germination_centers_by_roi(
    boxes: list[tuple[str, float, tuple[int, int, int, int]]],
    roi: tuple[int, int, int, int] | None,
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    if roi is None:
        return boxes

    rx1, ry1, rx2, ry2 = roi
    kept: list[tuple[str, float, tuple[int, int, int, int]]] = []
    removed = 0
    for item in boxes:
        cls_name, _conf, bbox = item
        if cls_name not in GERMINATION_CLASSES:
            kept.append(item)
            continue
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
            kept.append(item)
        else:
            removed += 1
    if removed:
        print(f"  [germ] {removed} box(es) fora da área útil da grade removida(s)")
    return kept


def _count_visible_cells_by_grid(img_bgr: np.ndarray) -> Optional[int]:
    """Conta células por linhas da grade branca quando a bandeja está visível."""
    h, w = img_bgr.shape[:2]
    bright_grid, vertical_bands, horizontal_bands = _grid_bands_from_image(img_bgr)
    cols = _count_axis_slots(vertical_bands, w, bright_grid, "x", horizontal_bands)
    rows = _count_axis_slots(horizontal_bands, h, bright_grid, "y", vertical_bands)
    if cols is None or rows is None:
        return None

    count = cols * rows
    if not (4 <= count <= 500):
        return None

    print(
        f"  [cells] Grade detectada: {cols} colunas x {rows} linhas = {count} células visíveis"
    )
    return count


def _count_visible_cells_by_contours(img_bgr: np.ndarray) -> Optional[int]:
    """Conta células visíveis por contornos escuros do substrato."""
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


def _count_visible_cells_with_method(
    img_bgr: np.ndarray,
    image_quality: Optional[dict] = None,
) -> tuple[Optional[int], str | None]:
    """Conta células visíveis da bandeja e informa qual sinal venceu.

    Em luz neutra, os contornos escuros do substrato costumam representar bem
    células completas. Em LED roxo/magenta, esses contornos se fundem; nesse
    caso a geometria das linhas claras da grade é um sinal mais estável.
    """
    if image_quality is None:
        image_quality = _assess_image_quality(img_bgr)

    contour_count = _count_visible_cells_by_contours(img_bgr)
    prefer_grid = image_quality.get("issue") in {"led_purple", "led_magenta"}

    grid_count = _count_visible_cells_by_grid(img_bgr)

    if not prefer_grid and contour_count is not None:
        if grid_count is not None and abs(grid_count - contour_count) <= max(2, contour_count * 0.20):
            print(f"  [cells] Contornos={contour_count}; usando grade={grid_count}")
            return grid_count, "grid"
        return contour_count, "contours"

    if prefer_grid and grid_count is not None:
        if contour_count is not None and abs(grid_count - contour_count) >= max(4, contour_count * 0.35):
            print(f"  [cells] Contornos={contour_count}; usando grade={grid_count}")
        return grid_count, "grid"

    if contour_count is not None:
        return contour_count, "contours"

    return grid_count, "grid" if grid_count is not None else None


def _count_visible_cells(
    img_bgr: np.ndarray,
    image_quality: Optional[dict] = None,
) -> Optional[int]:
    count, _method = _count_visible_cells_with_method(img_bgr, image_quality)
    return count


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
    raw_method: str | None = None,
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
    # células vazias. Quando veio da grade real, empate pode ser bandeja 100%.
    if raw_detected is not None:
        min_plausible = max(2, germinated_count if raw_method == "grid" else germinated_count + 1)
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


def _cells_warning_message(
    raw_detected: Optional[int],
    germinated_count: int,
    cells_origin: str,
) -> str | None:
    if cells_origin != "fallback_default":
        return None
    if raw_detected is not None and raw_detected <= germinated_count:
        return (
            "_💡 A foto parece ser um recorte pequeno ou com células cortadas. "
            "Localizei as plantas visíveis, mas não usei essa grade incompleta para calcular taxa. "
            "Para taxa confiável, envie a bandeja inteira ou informe o total de células na legenda (ex: '128')._"
        )
    return (
        "_💡 Não consegui confirmar as células visíveis com segurança. Para taxa confiável, "
        "envie a bandeja inteira ou informe o total de células na legenda (ex: '128')._"
    )


def _nms_germination_boxes(
    boxes: list[tuple[str, float, tuple[int, int, int, int]]],
    iou_threshold: float = 0.50,
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """Non-Max Suppression em germinacoes.

    Mantem o box de MAIOR area quando 2+ se sobrepoem com IoU > threshold.
    Preserva todos os boxes que nao sao Germinacao.
    """
    germ_items = [item for item in boxes if item[0] in GERMINATION_CLASSES]
    other_items = [item for item in boxes if item[0] not in GERMINATION_CLASSES]

    if len(germ_items) < 2:
        return boxes

    def iou(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> float:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
        area2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
        union = area1 + area2 - inter
        return inter / max(union, 1)

    germ_sorted = sorted(
        germ_items,
        key=lambda item: -((item[2][2] - item[2][0]) * (item[2][3] - item[2][1])),
    )
    kept: list[tuple[str, float, tuple[int, int, int, int]]] = []
    suppressed = 0
    for item in germ_sorted:
        if any(iou(item[2], kept_item[2]) > iou_threshold for kept_item in kept):
            suppressed += 1
            continue
        kept.append(item)

    if suppressed:
        print(f"  [germ] NMS removeu {suppressed} box(es) sobreposto(s) (IoU>{iou_threshold})")
    return other_items + kept


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
    image_quality = _assess_image_quality(img_bgr)
    magenta_mode = image_quality.get("issue") == "led_magenta"
    # SAHI ajuda em fotos grandes, mas em recortes pequenos (ex: 512x512)
    # tende a gerar caixas largas atravessando várias células.
    use_sahi = _SAHI_AVAILABLE and min(h, w) >= 640
    print(f"  Inferencia {'SAHI (tiles)' if use_sahi else 'direta'} em {w}x{h}")
    if image_quality.get("warning"):
        print(f"  [quality] {image_quality['level']}: {image_quality['issue']}")

    # Normaliza iluminação para inferência (original preservado para anotação visual)
    img_for_inference = _enhance_for_yolo(img_bgr, image_quality)

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
            germ_conf = max(0.30, conf_threshold + 0.05) if magenta_mode else max(0.12, conf_threshold - 0.13)
            folha_conf = max(0.20, conf_threshold - 0.05) if magenta_mode else max(0.15, conf_threshold - 0.10)
            raw_boxes = [
                d for d in raw_boxes
                if (d["cls_name"] == "Germinacao" and d["conf"] >= germ_conf)
                or (d["cls_name"] == "Folha" and d["conf"] >= folha_conf)
                or (d["cls_name"] not in ("Germinacao", "Folha") and d["conf"] >= conf_threshold)
            ]
        else:
            # Usa conf mais baixo no predict para capturar Germinacoes e Folhas periféricas
            folha_conf = max(0.20, conf_threshold - 0.05) if magenta_mode else max(0.15, conf_threshold - 0.10)
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
            germ_conf = max(0.30, conf_threshold + 0.05) if magenta_mode else max(0.12, conf_threshold - 0.13)
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
    signal_img_bgr = img_for_inference if magenta_mode else img_bgr
    issue = image_quality.get("issue")

    # leaf_based depende de YOLO confiavel de Folha, so funciona em luz normal.
    if issue is None:
        clamped_boxes = _leaf_based_germination_fallback(img_bgr, clamped_boxes, w, h)

    # green_component: aglomerados verdes grandes em LED. Desligado por padrao
    # desde o modelo de 2026-05-26 porque gera bboxes-blob enormes englobando
    # varias plantas. Reativar com GERMINAVISION_GREEN_COMPONENT_FALLBACK=1.
    if issue in {"led_purple", "led_magenta"} and os.environ.get("GERMINAVISION_GREEN_COMPONENT_FALLBACK") == "1":
        clamped_boxes = _green_component_germination_fallback(img_bgr, clamped_boxes, image_quality)

    # tiny_green em luz normal gera falsos positivos em residuos verdes;
    # em LED roxo/magenta ajuda a recuperar mudas pequenas escuras.
    if issue in {"led_purple", "led_magenta"}:
        clamped_boxes = _tiny_green_germination_fallback(img_bgr, clamped_boxes, image_quality)

    # Hybrid intra-cell so em magenta. Desligado por padrao desde o modelo de
    # 2026-05-26 porque a heuristica cluster-por-cell empilhava bboxes em
    # plantas adultas sobrepostas. Reativar com GERMINAVISION_HYBRID_FALLBACK=1.
    if issue == "led_magenta" and os.environ.get("GERMINAVISION_HYBRID_FALLBACK") == "1":
        clamped_boxes = _hybrid_grid_cluster_fallback(img_bgr, clamped_boxes, image_quality)

    # cluster_based fica apenas no LED roxo, onde o sinal vegetal e medio.
    if issue == "led_purple":
        clamped_boxes = _cluster_based_germination_fallback(img_bgr, clamped_boxes, image_quality)
    if magenta_mode:
        clamped_boxes = _filter_germination_centers_by_roi(clamped_boxes, _magenta_grid_roi(img_bgr))
    clamped_boxes = _filter_germination_by_green_signal(signal_img_bgr, clamped_boxes)
    clamped_boxes = _nms_germination_boxes(clamped_boxes, iou_threshold=0.50)
    clamped_boxes = _split_tall_germination_boxes(signal_img_bgr, clamped_boxes)
    clamped_boxes = _dedupe_germination_boxes(clamped_boxes)
    clamped_boxes = _nms_germination_boxes(clamped_boxes, iou_threshold=0.50)
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
    # Anotacao sempre usa a imagem original (preserva cor real, mesmo em magenta).
    # Para voltar a visualizacao pseudo-natural antiga, exporte
    # GERMINAVISION_NATURALIZE_MAGENTA=1.
    if magenta_mode and os.environ.get("GERMINAVISION_NATURALIZE_MAGENTA") == "1":
        img_annotated = _naturalize_magenta_for_display(img_bgr, img_for_inference, germ_boxes)
    else:
        img_annotated = img_bgr.copy()

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
            signal_crop = signal_img_bgr[y1:y2, x1:x2]
            leaf_contour = _estimate_leaves_by_contours(signal_crop)
            # Usa o maior sinal: YOLO pode subestimar (não detectou todas as Folhas),
            # contorno pode subestimar (threshold colapsou peaks sobrepostos)
            leaf_n = max(leaf_yolo, leaf_contour)
            if leaf_n <= 0 and signal_crop.size:
                crop_hsv = cv2.cvtColor(signal_crop, cv2.COLOR_BGR2HSV)
                if _plant_signal_ratio(crop_hsv) >= 0.18:
                    leaf_n = 1
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

    raw_detected, raw_detected_method = _count_visible_cells_with_method(img_bgr, image_quality)
    detected_cells, cells_origin = _resolve_cell_count(
        raw_detected, germinated_count, tray_capacity_override, raw_detected_method
    )
    cells_warning = _cells_warning_message(raw_detected, germinated_count, cells_origin)

    germination_rate = round(germinated_count / detected_cells * 100, 1) if detected_cells > 0 else 0.0
    leaf_avg = round(sum(leaf_counts) / len(leaf_counts), 1) if leaf_counts else 0.0
    elapsed = round(time.time() - t0, 2)
    rate_scope = {
        "caption": "tray",
        "detected_visible": "visible_area",
        "fallback_default": "default_capacity",
    }.get(cells_origin, "visible_area")

    rate_reliable = cells_origin != "fallback_default" and image_quality.get("level") != "low"

    return {
        "total_detected":   total,
        "germinated":       germinated_count,
        "germination_rate": germination_rate,
        "cells_detected":   detected_cells,
        "cells_origin":     cells_origin,
        "cells_warning":    cells_warning,
        "rate_reliable":    rate_reliable,
        "rate_scope":       rate_scope,
        "image_quality":    image_quality,
        "quality_level":    image_quality.get("level"),
        "quality_warning":  image_quality.get("warning"),
        "leaf_avg":         leaf_avg,
        "total_folhas_estimadas": int(round(leaf_avg * germinated_count)),
        "leaf_counts":      leaf_counts,
        "tray_capacity":    TRAY_CAPACITY,
        "detections":       detections,
        "result_image":     f"/static/results/{result_name}",
        "inference_time_s": elapsed,
    }
