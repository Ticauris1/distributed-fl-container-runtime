"""Validate SDNET2018 pilot pairings, masks, and group-safe splits."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def validate(workspace: Path) -> None:
    workspace = workspace.expanduser().resolve()
    with (workspace / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Manifest contains no samples")
    groups = defaultdict(set)
    status_counts = Counter()
    class_counts = Counter()
    for row in rows:
        image_path = workspace / row["image_path"]
        mask_path = workspace / row["mask_path"]
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing pair for {row['sample_id']}")
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                raise ValueError(f"Size mismatch for {row['sample_id']}")
            values = set(np.unique(np.asarray(mask.convert("L"))).tolist())
            if not values.issubset({0, 255}):
                raise ValueError(f"Non-binary mask for {row['sample_id']}: {values}")
            if row["is_cracked"] == "0" and values != {0}:
                raise ValueError(f"Uncracked sample has foreground pixels: {row['sample_id']}")
        groups[row["source_group_id"]].add(row["split"])
        status_counts[row["mask_status"]] += 1
        class_counts[(row["split"], row["surface"], row["class_label"])] += 1
    leaks = {group: splits for group, splits in groups.items() if len(splits) > 1}
    if leaks:
        raise ValueError(f"Original-photo group leakage detected: {leaks}")
    print(f"PASS: {len(rows)} paired samples")
    print(f"PASS: {len(groups)} original-photo groups stay within one split")
    print("Mask status counts:")
    for key, count in sorted(status_counts.items()):
        print(f"  {key}: {count}")
    print("Split/surface/class counts:")
    for key, count in sorted(class_counts.items()):
        print(f"  {key}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace", type=Path, default=Path(__file__).resolve().parent / "workspace"
    )
    validate(parser.parse_args().workspace)
