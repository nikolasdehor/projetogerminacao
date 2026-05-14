"""
Pipeline de preparação do dataset de morango:

1. Lê PNGs anotados em LabelMe (.png + .json) de seedling.v1i.yolov8.bak.2026-05-13/train/images/<data>/
2. Renomeia com prefixo de data para evitar colisão (ex: 28-01-2025_1A.png)
3. Fatia cada PNG em 8 tiles (grade 4x2)
4. Converte polígonos LabelMe → bbox YOLO normalizada por tile
5. Faz split 70/20/10 em train/valid/test
6. Gera data.yaml com 2 classes [Germinacao, Folha]

Uso:
    python prepare_dataset.py
"""
from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────

SRC_ROOT = Path("_source_pngs")           # pasta com PNGs+JSONs LabelMe nas subpastas datadas
DST_ROOT = Path("dataset")                 # output YOLO já com split train/valid/test
TILES_X, TILES_Y = 4, 2          # grade 4 colunas × 2 linhas = 8 tiles
CLASSES = ["Germinacao", "Folha"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
SPLIT_RATIOS = (0.70, 0.20, 0.10)  # train, valid, test
MIN_BBOX_AREA_PX = 25            # descarta bboxes minúsculos (ruído após corte)
SEED = 42


def labelme_to_tile_yolo(
    shapes: list[dict],
    tile_x0: int, tile_y0: int, tile_w: int, tile_h: int
) -> list[str]:
    """
    Para cada shape do LabelMe que intersecta o tile, gera linha YOLO recortada ao tile.
    Estratégia: usa bbox envoltória do polígono, recorta ao tile, descarta se vazia/muito pequena.
    """
    lines: list[str] = []
    for shape in shapes:
        label = shape.get("label", "")
        if label not in CLASS_TO_ID:
            continue

        points = shape.get("points", [])
        if len(points) < 2:
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        # bbox no sistema da imagem original
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Recorta ao tile
        cx_min = max(x_min, tile_x0)
        cy_min = max(y_min, tile_y0)
        cx_max = min(x_max, tile_x0 + tile_w)
        cy_max = min(y_max, tile_y0 + tile_h)

        if cx_max <= cx_min or cy_max <= cy_min:
            continue  # bbox fora do tile

        w_clipped = cx_max - cx_min
        h_clipped = cy_max - cy_min

        if w_clipped * h_clipped < MIN_BBOX_AREA_PX:
            continue  # bbox residual minúsculo

        # Coordenadas relativas ao tile, normalizadas
        cx = ((cx_min + cx_max) / 2 - tile_x0) / tile_w
        cy = ((cy_min + cy_max) / 2 - tile_y0) / tile_h
        nw = w_clipped / tile_w
        nh = h_clipped / tile_h

        cls_id = CLASS_TO_ID[label]
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def slice_png(img: Image.Image, base_stem: str, shapes: list[dict]) -> list[tuple[str, Image.Image, list[str]]]:
    """Fatia uma imagem em TILES_X×TILES_Y tiles e retorna (nome_base, tile_img, yolo_lines) para cada."""
    W, H = img.size
    tile_w = W // TILES_X
    tile_h = H // TILES_Y
    out: list[tuple[str, Image.Image, list[str]]] = []
    for ty in range(TILES_Y):
        for tx in range(TILES_X):
            x0 = tx * tile_w
            y0 = ty * tile_h
            crop = img.crop((x0, y0, x0 + tile_w, y0 + tile_h))
            tile_id = ty * TILES_X + tx  # 0..7
            tile_stem = f"{base_stem}_t{tile_id}"
            lines = labelme_to_tile_yolo(shapes, x0, y0, tile_w, tile_h)
            out.append((tile_stem, crop, lines))
    return out


def main() -> None:
    if not SRC_ROOT.exists():
        raise SystemExit(f"Pasta fonte não existe: {SRC_ROOT}")

    # Coleta todos PNGs nas subpastas datadas
    samples: list[tuple[Path, Path | None, str]] = []  # (png_path, json_path|None, base_stem)
    date_dirs = sorted([d for d in SRC_ROOT.iterdir() if d.is_dir()])
    if not date_dirs:
        raise SystemExit(f"Nenhuma subpasta datada em {SRC_ROOT}")

    for d in date_dirs:
        for png in sorted(d.glob("*.png")):
            jpath = png.with_suffix(".json")
            # Renomeia com prefixo da data se nome ainda não tem
            date_prefix = d.name  # ex: "28-01-2025"
            if not png.stem.startswith(date_prefix):
                base_stem = f"{date_prefix}_{png.stem}"
            else:
                base_stem = png.stem
            samples.append((png, jpath if jpath.exists() else None, base_stem))

    print(f"📦 Total de PNGs encontrados: {len(samples)}")
    annotated = sum(1 for _, j, _ in samples if j is not None)
    print(f"   Anotados (com .json): {annotated}")
    print(f"   Sem anotação: {len(samples) - annotated}")

    # Embaralha e faz split por imagem ORIGINAL (não por tile) para evitar leakage
    random.seed(SEED)
    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * SPLIT_RATIOS[0])
    n_valid = int(n * SPLIT_RATIOS[1])
    splits = {
        "train": samples[:n_train],
        "valid": samples[n_train:n_train + n_valid],
        "test":  samples[n_train + n_valid:],
    }
    for k, v in splits.items():
        print(f"   Split {k}: {len(v)} imagens")

    # Processa cada split
    class_counter: Counter = Counter()
    tiles_written = 0
    tiles_empty = 0

    for split_name, items in splits.items():
        img_dir = DST_ROOT / split_name / "images"
        lbl_dir = DST_ROOT / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for png_path, json_path, base_stem in items:
            img = Image.open(png_path).convert("RGB")
            shapes: list[dict] = []
            if json_path is not None:
                with open(json_path) as f:
                    shapes = json.load(f).get("shapes", [])

            for tile_stem, tile_img, yolo_lines in slice_png(img, base_stem, shapes):
                # Sempre salva imagem (mesmo sem labels — útil para val/test)
                tile_img.save(img_dir / f"{tile_stem}.jpg", quality=92)
                # Label .txt: se houver linhas, escreve; se vazio, ainda cria arquivo vazio para YOLO
                lbl_path = lbl_dir / f"{tile_stem}.txt"
                lbl_path.write_text("\n".join(yolo_lines))
                tiles_written += 1
                if not yolo_lines:
                    tiles_empty += 1
                for line in yolo_lines:
                    cls_id = int(line.split()[0])
                    class_counter[CLASSES[cls_id]] += 1

    # Gera data.yaml
    yaml_text = (
        f"# Dataset Morango — Germinação (gerado por prepare_dataset.py)\n"
        f"path: {DST_ROOT.absolute()}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )
    (DST_ROOT / "data.yaml").write_text(yaml_text)

    # Relatório final
    print("\n✅ Preparação concluída")
    print(f"   Tiles gerados: {tiles_written}")
    print(f"   Tiles vazios (sem objetos): {tiles_empty}")
    print(f"   Distribuição de classes:")
    for cls, n in class_counter.most_common():
        print(f"     {cls}: {n}")
    print(f"\n   data.yaml escrito em {DST_ROOT / 'data.yaml'}")


if __name__ == "__main__":
    main()
