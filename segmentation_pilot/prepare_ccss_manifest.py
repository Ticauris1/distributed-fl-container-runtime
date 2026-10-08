"""Build a project manifest from the official CCSS benchmark split."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image


FIELDS = (
    "sample_id",
    "split",
    "image_path",
    "mask_path",
    "is_cracked",
    "source_group_id",
    "mask_status",
    "dataset_source",
)


def build_manifest(dataset_root: Path, output: Path) -> Path:
    dataset_root = dataset_root.expanduser().resolve()
    benchmark = dataset_root / "benchmarkingDatasets"
    split_roots = {
        "train": benchmark / "train" / "noAugmentation-noPatch",
        "val": benchmark / "validation",
        "test": benchmark / "test",
    }
    rows = []
    for split, split_root in split_roots.items():
        image_dir, mask_dir = split_root / "img", split_root / "lbl"
        images = {path.name: path for path in image_dir.glob("*.jpg")}
        masks = {path.name: path for path in mask_dir.glob("*.jpg")}
        if images.keys() != masks.keys():
            raise ValueError(
                f"CCSS image/mask mismatch for {split}: "
                f"image_only={sorted(images.keys() - masks.keys())[:5]}, "
                f"mask_only={sorted(masks.keys() - images.keys())[:5]}"
            )
        for filename in sorted(images):
            image_path, mask_path = images[filename], masks[filename]
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                if image.size != mask.size:
                    raise ValueError(
                        f"CCSS size mismatch for {filename}: {image.size} vs {mask.size}"
                    )
                is_cracked = bool(
                    (np.asarray(mask.convert("L"), dtype=np.uint8) >= 128).any()
                )
            source = "ccss" if not image_path.stem.isdigit() else "external_458"
            rows.append({
                "sample_id": f"ccss_{split}_{image_path.stem}",
                "split": split,
                "image_path": str(image_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "is_cracked": str(is_cracked),
                "source_group_id": f"{source}_{image_path.stem}",
                "mask_status": "approved_manual",
                "dataset_source": source,
            })

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    counts = {split: sum(row["split"] == split for row in rows) for split in split_roots}
    print(f"CCSS manifest: {output}")
    print(f"Split counts: {counts}")
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            os.environ.get(
                "CCSS_DATASET_ROOT",
                "data/CCSS-DATA-V4-withVal",
            )
        ),
        help="Root of the extracted CCSS-DATA-V4-withVal download.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "ccss_benchmark_manifest.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    build_manifest(arguments.dataset_root, arguments.output)
