"""Create a leakage-safe, fixed geometric expansion of a segmentation manifest."""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


GEOMETRIC_VARIANTS = {
    "hflip_rot90": (
        Image.Transpose.FLIP_LEFT_RIGHT,
        Image.Transpose.ROTATE_90,
    ),
    "vflip_rot270": (
        Image.Transpose.FLIP_TOP_BOTTOM,
        Image.Transpose.ROTATE_270,
    ),
}


def _transform(source: Path, destination: Path, operations) -> None:
    with Image.open(source) as opened:
        transformed = opened.copy()
    for operation in operations:
        transformed = transformed.transpose(operation)
    destination.parent.mkdir(parents=True, exist_ok=True)
    transformed.save(destination)


def build_dataset(source_manifest: Path, output_root: Path) -> Path:
    source_manifest = source_manifest.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    manifest_path = output_root / "manifest.csv"
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output is not empty: {output_root}. Choose a new output directory."
        )
    with source_manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
        fieldnames = list(rows[0])
    source_root = source_manifest.parent
    output_rows = []
    for row in rows:
        split = row["split"]
        image_source = source_root / row["image_path"]
        mask_source = source_root / row["mask_path"]
        image_relative = Path(split) / "images" / f"{row['sample_id']}{image_source.suffix.lower()}"
        mask_relative = Path(split) / "masks" / f"{row['sample_id']}{mask_source.suffix.lower()}"
        image_destination = output_root / image_relative
        mask_destination = output_root / mask_relative
        image_destination.parent.mkdir(parents=True, exist_ok=True)
        mask_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_source, image_destination)
        shutil.copy2(mask_source, mask_destination)
        copied = dict(row)
        copied["image_path"] = image_relative.as_posix()
        copied["mask_path"] = mask_relative.as_posix()
        output_rows.append(copied)
        if split != "train":
            continue
        for suffix, operations in GEOMETRIC_VARIANTS.items():
            augmented = dict(row)
            augmented["sample_id"] = f"{row['sample_id']}__{suffix}"
            image_relative = Path("train/images") / f"{augmented['sample_id']}{image_source.suffix.lower()}"
            mask_relative = Path("train/masks") / f"{augmented['sample_id']}.png"
            augmented["image_path"] = image_relative.as_posix()
            augmented["mask_path"] = mask_relative.as_posix()
            if "source_path" in augmented:
                augmented["source_path"] = f"{row.get('source_path', image_source)}#{suffix}"
            _transform(image_source, output_root / image_relative, operations)
            _transform(mask_source, output_root / mask_relative, operations)
            output_rows.append(augmented)
    output_root.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    counts = Counter(row["split"] for row in output_rows)
    print(f"Fixed geometric dataset: {manifest_path}")
    print(f"Rows by split: {dict(counts)}")
    return manifest_path


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=project_root / "segmentation_pilot/workspace/manifest.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "segmentation_pilot/workspace_fixed_geometric",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    build_dataset(arguments.source_manifest, arguments.output_root)
