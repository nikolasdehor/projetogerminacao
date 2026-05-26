"""Triagem automatica das auto-labels de uploads do WhatsApp.

Regras:
- avg_conf >= 0.5 AND n_germ >= 3: aceita direto.
- 0.35 <= avg_conf < 0.5: re-roda inferencia com threshold maior e filtra
  caixas claramente fora da grade detectada.
- avg_conf < 0.35 OU anomalias graves: rejeita.
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from app.inference import (
    _assess_image_quality,
    _detect_grid_via_edges,
    _magenta_grid_roi,
    load_model,
    run_inference,
)

ROOT = Path(__file__).resolve().parents[1]
AUTOLABEL = ROOT / "dataset" / "autolabel_uploads"
META_IN = AUTOLABEL / "metadata.csv"
META_OUT = AUTOLABEL / "metadata_curated.csv"
ACCEPTED_DIR = AUTOLABEL / "accepted"
REJECTED_DIR = AUTOLABEL / "rejected"

for base_dir in (ACCEPTED_DIR, REJECTED_DIR):
    (base_dir / "images").mkdir(exist_ok=True, parents=True)
    (base_dir / "labels").mkdir(exist_ok=True, parents=True)

CLASS_NAMES = {0: "Germinacao", 1: "Folha"}


def xyxy_to_yolo(box: tuple[float, float, float, float], w: int, h: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)


def _copy_label_if_exists(src_lbl: Path, dst_dir: Path) -> None:
    if src_lbl.exists():
        shutil.copy(src_lbl, dst_dir / src_lbl.name)


def _source_has_severe_anomaly(label_path: Path) -> bool:
    if not label_path.exists():
        return False

    for raw in label_path.read_text().splitlines():
        parts = raw.split()
        if len(parts) != 5:
            return True
        try:
            _cls, cx, cy, bw, bh = parts
            cx_f, cy_f, bw_f, bh_f = map(float, (cx, cy, bw, bh))
        except ValueError:
            return True

        if not (0 <= cx_f <= 1 and 0 <= cy_f <= 1 and 0 < bw_f <= 1 and 0 < bh_f <= 1):
            return True
        if bw_f * bh_f > 0.45:
            return True

    return False


def _grid_roi(img: np.ndarray, quality: dict) -> tuple[int, int, int, int] | None:
    h, w = img.shape[:2]

    if quality.get("issue") == "led_magenta":
        roi = _magenta_grid_roi(img)
        if roi is not None:
            return roi

    grid = _detect_grid_via_edges(img, quality)
    if grid is None:
        return None

    vertical_xs, horizontal_ys = grid
    if len(vertical_xs) < 3 or len(horizontal_ys) < 3:
        return None

    def _margin(points: list[int], fallback: int) -> int:
        diffs = np.diff(sorted(points))
        usable = diffs[diffs > 4]
        if usable.size == 0:
            return fallback
        return max(12, int(float(np.median(usable)) * 0.65))

    mx = _margin(vertical_xs, int(w * 0.05))
    my = _margin(horizontal_ys, int(h * 0.05))
    return (
        max(0, min(vertical_xs) - mx),
        max(0, min(horizontal_ys) - my),
        min(w, max(vertical_xs) + mx),
        min(h, max(horizontal_ys) + my),
    )


def _box_inside_roi(box: tuple[float, float, float, float], roi: tuple[int, int, int, int] | None) -> bool:
    if roi is None:
        return True

    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = roi
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def _valid_xyxy(box: tuple[float, float, float, float], w: int, h: int) -> bool:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1 or bh <= 1:
        return False
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return False
    if (bw * bh) / max(w * h, 1) > 0.45:
        return False
    return True


def main() -> None:
    if not META_IN.exists():
        raise SystemExit(f"Metadata nao encontrado: {META_IN}")

    model = load_model(str(ROOT / "models" / "best.pt"))
    rows_in = list(csv.DictReader(META_IN.open()))
    rows_out: list[dict[str, object]] = []

    accepted = 0
    corrected = 0
    rejected = 0

    for r in rows_in:
        fname = r["filename"]
        avg_conf = float(r["avg_confidence"])
        n_germ = int(r["n_germinations"])
        src_img = AUTOLABEL / "images" / fname
        src_lbl = AUTOLABEL / "labels" / Path(fname).with_suffix(".txt").name

        if not src_img.exists():
            continue

        if _source_has_severe_anomaly(src_lbl):
            shutil.copy(src_img, REJECTED_DIR / "images" / fname)
            _copy_label_if_exists(src_lbl, REJECTED_DIR / "labels")
            rejected += 1
            r["curation"] = "rejected_anomaly"
            r["final_n_germ"] = 0
            rows_out.append(r)
            continue

        if avg_conf >= 0.5 and n_germ >= 3:
            shutil.copy(src_img, ACCEPTED_DIR / "images" / fname)
            _copy_label_if_exists(src_lbl, ACCEPTED_DIR / "labels")
            accepted += 1
            r["curation"] = "accepted"
            r["final_n_germ"] = n_germ
            rows_out.append(r)
            continue

        if avg_conf < 0.35:
            shutil.copy(src_img, REJECTED_DIR / "images" / fname)
            _copy_label_if_exists(src_lbl, REJECTED_DIR / "labels")
            rejected += 1
            r["curation"] = "rejected_low_conf"
            r["final_n_germ"] = 0
            rows_out.append(r)
            continue

        img = cv2.imread(str(src_img))
        if img is None:
            continue

        h, w = img.shape[:2]
        quality = _assess_image_quality(img)
        roi = _grid_roi(img, quality)

        try:
            res = run_inference(
                str(src_img),
                model,
                "/tmp",
                conf_threshold=0.40,
                class_names=CLASS_NAMES,
            )
        except Exception as exc:
            print(f"  ERR {fname}: {exc}")
            shutil.copy(src_img, REJECTED_DIR / "images" / fname)
            _copy_label_if_exists(src_lbl, REJECTED_DIR / "labels")
            rejected += 1
            r["curation"] = "rejected_error"
            r["final_n_germ"] = 0
            rows_out.append(r)
            continue

        new_lines = []
        new_germ = 0
        new_folha = 0
        for d in res.get("detections", []):
            cls = d["class"]
            cls_id = 0 if cls == "Germinacao" else (1 if cls == "Folha" else None)
            if cls_id is None:
                continue

            x1, y1, x2, y2 = d["bbox"]
            box = (float(x1), float(y1), float(x2), float(y2))
            if not _valid_xyxy(box, w, h):
                continue
            if not _box_inside_roi(box, roi):
                continue

            cx, cy, bw, bh = xyxy_to_yolo(box, w, h)
            new_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            if cls_id == 0:
                new_germ += 1
            else:
                new_folha += 1

        if new_germ < 3:
            shutil.copy(src_img, REJECTED_DIR / "images" / fname)
            (REJECTED_DIR / "labels" / src_lbl.name).write_text("\n".join(new_lines))
            rejected += 1
            r["curation"] = "rejected_after_recheck"
            r["final_n_germ"] = new_germ
            rows_out.append(r)
            continue

        shutil.copy(src_img, ACCEPTED_DIR / "images" / fname)
        (ACCEPTED_DIR / "labels" / src_lbl.name).write_text("\n".join(new_lines))
        corrected += 1
        r["curation"] = "corrected"
        r["final_n_germ"] = new_germ
        r["final_n_folhas"] = new_folha
        rows_out.append(r)

    if rows_out:
        fieldnames: list[str] = []
        for row in rows_out:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with META_OUT.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_out)

    print(f"\nTotal: {len(rows_in)}")
    print(f"  accepted (conf alta):     {accepted}")
    print(f"  corrected (re-label):     {corrected}")
    print(f"  rejected:                 {rejected}")
    print(f"\nAccepted images: {ACCEPTED_DIR / 'images'}")
    print(f"Rejected images: {REJECTED_DIR / 'images'}")
    print(f"Metadata: {META_OUT}")


if __name__ == "__main__":
    main()
