"""Auto-label WhatsApp uploads for YOLO retraining.

Runs the current modal pipeline (best.pt plus quality-based fallbacks) for each
upload and writes YOLO-format labels under dataset/autolabel_uploads/.

Output:
- dataset/autolabel_uploads/images/<upload_name>.jpeg
- dataset/autolabel_uploads/labels/<upload_name>.txt
- dataset/autolabel_uploads/metadata.csv
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from app.inference import _assess_image_quality, load_model, run_inference

ROOT = Path(__file__).resolve().parents[1]
UPLOADS = ROOT / "static" / "uploads"
OUT_DIR = ROOT / "dataset" / "autolabel_uploads"
IMG_DIR = OUT_DIR / "images"
LBL_DIR = OUT_DIR / "labels"
META_CSV = OUT_DIR / "metadata.csv"

IMG_DIR.mkdir(parents=True, exist_ok=True)
LBL_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: "Germinacao", 1: "Folha"}


def xyxy_to_yolo(box: tuple[int, int, int, int], w: int, h: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh


def main() -> None:
    model = load_model(str(ROOT / "models" / "best.pt"))
    uploads = sorted(UPLOADS.glob("wa_*.jpeg"))
    print(f"Auto-labeling {len(uploads)} uploads...")

    rows = []
    for src in uploads:
        img = cv2.imread(str(src))
        if img is None:
            print(f"  SKIP {src.name}: imagem invalida")
            continue

        h, w = img.shape[:2]
        quality = _assess_image_quality(img)
        try:
            res = run_inference(
                str(src),
                model,
                "/tmp",
                conf_threshold=0.25,
                class_names=CLASS_NAMES,
            )
        except Exception as exc:
            print(f"  ERR {src.name}: {exc}")
            continue

        dst_img = IMG_DIR / src.name
        shutil.copy(src, dst_img)

        lines = []
        confs = []
        n_germ = 0
        n_folha = 0
        for detection in res.get("detections", []):
            x1, y1, x2, y2 = detection["bbox"]
            cls = detection["class"]
            cls_id = 0 if cls == "Germinacao" else (1 if cls == "Folha" else None)
            if cls_id is None:
                continue

            cx, cy, bw, bh = xyxy_to_yolo((x1, y1, x2, y2), w, h)
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            confs.append(detection.get("confidence", 0.0))
            if cls_id == 0:
                n_germ += 1
            else:
                n_folha += 1

        lbl_path = LBL_DIR / src.with_suffix(".txt").name
        lbl_path.write_text("\n".join(lines))

        avg_conf = sum(confs) / max(len(confs), 1)
        row = {
            "filename": src.name,
            "issue": quality.get("issue") or "none",
            "n_germinations": n_germ,
            "n_folhas": n_folha,
            "avg_confidence": round(avg_conf, 3),
            "needs_review": "yes" if avg_conf < 0.5 or n_germ < 3 else "no",
        }
        rows.append(row)
        print(
            f"  {src.name}: issue={row['issue']} "
            f"germ={n_germ} folha={n_folha} conf={avg_conf:.2f}"
        )

    with META_CSV.open("w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nLabels salvos em: {LBL_DIR}")
    print(f"Metadata: {META_CSV}")
    print(f"Total: {len(rows)} fotos auto-labeladas")
    print(f"Pra revisar (low confidence): {sum(1 for r in rows if r['needs_review'] == 'yes')}")


if __name__ == "__main__":
    main()
