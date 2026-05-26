"""Build an HTML dashboard for auto-label triage."""
from __future__ import annotations

import csv
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
AUTOLABEL = ROOT / "dataset" / "autolabel_uploads"
META = AUTOLABEL / "metadata.csv"
THUMBS_DIR = Path("/tmp/dataset_review_thumbs")
THUMBS_DIR.mkdir(exist_ok=True)
DASHBOARD = Path("/tmp/dataset_review.html")


def yolo_to_xyxy(cx: float, cy: float, bw: float, bh: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = int((cx - bw / 2) * width)
    y1 = int((cy - bh / 2) * height)
    x2 = int((cx + bw / 2) * width)
    y2 = int((cy + bh / 2) * height)
    return x1, y1, x2, y2


def draw_boxes(img_path: Path, lbl_path: Path, out_path: Path) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        return

    h, w = img.shape[:2]
    if lbl_path.exists():
        for line in lbl_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, w, h)
            color = (52, 211, 153) if cls == 0 else (36, 191, 251)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    max_dim = 600
    scale = max_dim / max(h, w)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 80])


def main() -> None:
    rows = []
    with META.open() as f:
        for row in csv.DictReader(f):
            rows.append(row)

    rows.sort(key=lambda row: (row["needs_review"] != "yes", float(row.get("avg_confidence", 0))))

    for row in rows:
        img_path = AUTOLABEL / "images" / row["filename"]
        lbl_path = AUTOLABEL / "labels" / Path(row["filename"]).with_suffix(".txt").name
        thumb_path = THUMBS_DIR / row["filename"]
        if img_path.exists():
            draw_boxes(img_path, lbl_path, thumb_path)

    html_parts = ["""<!doctype html>
<html><head><meta charset="utf-8">
<title>Dataset Review - GerminaVision</title>
<style>
body { font-family: -apple-system, sans-serif; background: #111; color: #eee; margin: 0; padding: 20px; }
h1 { margin-bottom: 10px; }
.stats { background: #222; padding: 12px; border-radius: 8px; margin-bottom: 20px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 20px; }
.card { background: #1a1a1a; border-radius: 8px; overflow: hidden; border: 2px solid transparent; }
.card.review { border-color: #f87171; }
.card.normal { border-color: #34d399; }
.card img { width: 100%; display: block; }
.meta { padding: 10px; font-size: 12px; }
.meta .filename { font-family: monospace; word-break: break-all; }
.badge { display: inline-block; padding: 2px 6px; border-radius: 4px; margin-right: 4px; font-size: 11px; }
.badge.magenta { background: #d946ef; }
.badge.purple { background: #7c3aed; }
.badge.normal { background: #10b981; }
.badge.mixed { background: #f59e0b; }
.badge.review { background: #ef4444; }
</style></head><body>
<h1>Dataset Review - 70 uploads auto-labeladas</h1>
"""]

    n_review = sum(1 for row in rows if row["needs_review"] == "yes")
    n_total = len(rows)
    by_issue: dict[str, int] = {}
    for row in rows:
        by_issue[row["issue"]] = by_issue.get(row["issue"], 0) + 1

    html_parts.append(f"""
<div class="stats">
<b>Total:</b> {n_total} fotos | <b>Precisam revisao:</b> {n_review} | <b>Por iluminacao:</b> {", ".join(f"{k}={v}" for k, v in by_issue.items())}
</div>
<div class="grid">
""")

    for row in rows:
        cls_card = "review" if row["needs_review"] == "yes" else "normal"
        issue = row["issue"]
        issue_badge_cls = (
            "magenta"
            if issue == "led_magenta"
            else ("purple" if issue == "led_purple" else ("normal" if issue == "none" else "mixed"))
        )
        review_badge = '<span class="badge review">REVISAR</span>' if row["needs_review"] == "yes" else ""
        html_parts.append(f"""
<div class="card {cls_card}">
<img src="dataset_review_thumbs/{row['filename']}" alt="{row['filename']}">
<div class="meta">
{review_badge}
<span class="badge {issue_badge_cls}">{issue}</span>
<div class="filename">{row['filename']}</div>
germ={row['n_germinations']} folha={row['n_folhas']} conf={row['avg_confidence']}
</div>
</div>
""")

    html_parts.append("</div></body></html>")
    DASHBOARD.write_text("".join(html_parts))
    print(f"Dashboard: {DASHBOARD}")
    print(f"Thumbs:    {THUMBS_DIR}")
    print(f"Abre: open {DASHBOARD}")


if __name__ == "__main__":
    main()
