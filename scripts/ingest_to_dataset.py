"""Adiciona fotos aceitas/corrigidas ao dataset com split 70/20/10.

Mantem dataset existente intacto, so adiciona novos arquivos com prefixo "wa_".
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACCEPTED = ROOT / "dataset" / "autolabel_uploads" / "accepted"
DATASET = ROOT / "dataset"
MANIFEST = ROOT / "dataset" / "autolabel_uploads" / "ingest_manifest.txt"

random.seed(42)


def main() -> None:
    images = sorted((ACCEPTED / "images").glob("wa_*"))
    if not images:
        print("Nenhuma imagem aceita pra ingerir")
        return

    random.shuffle(images)
    n = len(images)
    n_train = int(n * 0.70)
    n_valid = int(n * 0.20)
    train_set = images[:n_train]
    valid_set = images[n_train:n_train + n_valid]
    test_set = images[n_train + n_valid:]

    manifest_lines: list[str] = []

    for split, lst in [("train", train_set), ("valid", valid_set), ("test", test_set)]:
        (DATASET / split / "images").mkdir(exist_ok=True, parents=True)
        (DATASET / split / "labels").mkdir(exist_ok=True, parents=True)

        for img in lst:
            dst_img = DATASET / split / "images" / img.name
            dst_lbl = DATASET / split / "labels" / img.with_suffix(".txt").name
            src_lbl = ACCEPTED / "labels" / img.with_suffix(".txt").name

            shutil.copy(img, dst_img)
            manifest_lines.append(str(dst_img.relative_to(ROOT)))
            if src_lbl.exists():
                shutil.copy(src_lbl, dst_lbl)
                manifest_lines.append(str(dst_lbl.relative_to(ROOT)))

        print(f"{split}: +{len(lst)} fotos")

    MANIFEST.write_text("\n".join(manifest_lines) + "\n")
    print(f"\nDataset atualizado. Total: {n} fotos adicionadas (split 70/20/10)")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
