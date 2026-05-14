"""
Mistura dataset Roboflow (6 classes) com dataset morango (2 classes) atual.

Mapeamento Roboflow → 2 classes do morango:
  0 'askew'       → 0 Germinacao
  1 'noseedling'  → DESCARTA (sem objeto)
  2 'processed'   → 0 Germinacao
  3 'seedling'    → 0 Germinacao
  4 'twoseedling' → 0 Germinacao
  5 'weak'        → 0 Germinacao

Imagens Roboflow ganham prefixo 'rf_' para não colidir com tiles de morango.
Output sobrescreve dataset/ atual.
"""
from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

ROBOFLOW_SRC = Path("_roboflow")
DATASET_DST  = Path("dataset")
SPLITS       = ["train", "valid", "test"]

# Mapeamento: Roboflow id → novo id no dataset 2-class (None = descartar)
CLASS_MAP = {
    0: 0,    # askew → Germinacao
    1: None, # noseedling → descarta
    2: 0,    # processed → Germinacao
    3: 0,    # seedling → Germinacao
    4: 0,    # twoseedling → Germinacao
    5: 0,    # weak → Germinacao
}


def remap_label_file(src_txt: Path, dst_txt: Path) -> int:
    """Reescreve label .txt remapeando classe. Retorna count de linhas mantidas."""
    kept_lines = []
    if not src_txt.exists():
        dst_txt.write_text("")
        return 0
    for line in src_txt.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        old_cls = int(parts[0])
        new_cls = CLASS_MAP.get(old_cls)
        if new_cls is None:
            continue
        parts[0] = str(new_cls)
        kept_lines.append(" ".join(parts))
    dst_txt.write_text("\n".join(kept_lines))
    return len(kept_lines)


def main() -> None:
    if not ROBOFLOW_SRC.exists():
        raise SystemExit(f"{ROBOFLOW_SRC} não existe — extraia o ZIP primeiro")
    if not DATASET_DST.exists():
        raise SystemExit(f"{DATASET_DST} não existe — rode prepare_dataset.py primeiro")

    total_imgs = 0
    total_labels_kept = 0
    total_labels_dropped = 0
    class_counter: Counter = Counter()

    for split in SPLITS:
        src_img_dir = ROBOFLOW_SRC / split / "images"
        src_lbl_dir = ROBOFLOW_SRC / split / "labels"
        dst_img_dir = DATASET_DST / split / "images"
        dst_lbl_dir = DATASET_DST / split / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not src_img_dir.exists():
            print(f"  ⚠️  Pulando {split} — não existe no Roboflow")
            continue

        for img in sorted(src_img_dir.iterdir()):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            stem = img.stem
            new_name = f"rf_{stem}"

            # Copia imagem
            dst_img = dst_img_dir / f"{new_name}{img.suffix}"
            shutil.copy2(img, dst_img)

            # Remapeia label
            src_lbl = src_lbl_dir / f"{stem}.txt"
            dst_lbl = dst_lbl_dir / f"{new_name}.txt"
            kept = remap_label_file(src_lbl, dst_lbl)
            total_labels_kept += kept

            # Conta classes
            for ln in dst_lbl.read_text().strip().splitlines():
                if ln:
                    cls = int(ln.split()[0])
                    class_counter[cls] += 1

            # Conta labels descartados
            if src_lbl.exists():
                orig_lines = [l for l in src_lbl.read_text().strip().splitlines() if l]
                total_labels_dropped += len(orig_lines) - kept

            total_imgs += 1

    print("\n✅ Mix Roboflow → dataset/ concluído")
    print(f"   Imagens Roboflow adicionadas: {total_imgs}")
    print(f"   Labels mantidos (após remap): {total_labels_kept}")
    print(f"   Labels descartados (noseedling): {total_labels_dropped}")
    print(f"   Distribuição no que entrou:")
    names = {0: "Germinacao", 1: "Folha"}
    for cls_id, n in sorted(class_counter.items()):
        print(f"     {names.get(cls_id, cls_id)}: {n}")

    # Conta totais finais (Roboflow + morango já existente)
    print(f"\n   Totais finais no dataset/:")
    for split in SPLITS:
        n_img = len(list((DATASET_DST / split / "images").glob("*")))
        print(f"     {split}: {n_img} imagens")


if __name__ == "__main__":
    main()
