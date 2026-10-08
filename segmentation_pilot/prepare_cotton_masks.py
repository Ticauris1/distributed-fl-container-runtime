"""Create a group-safe cotton lesion-mask annotation workspace."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from mask_utils import zero_mask


MANIFEST_FIELDS = (
    "sample_id",
    "surface",
    "class_label",
    "is_cracked",
    "source_group_id",
    "split",
    "image_path",
    "mask_path",
    "mask_status",
    "source_path",
)
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def original_group_id(class_name: str, path: Path) -> str:
    """Group augmented derivatives with their apparent original image."""
    stem = re.split(r"_aug_\d+", path.stem, maxsplit=1)[0]
    stem = re.sub(r"\(\d+\)$", "", stem).strip()
    return f"class_{class_name}_{stem}"


def collect_records(dataset_root: Path):
    records = []
    class_dirs = sorted(
        (path for path in dataset_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    if not class_dirs:
        raise FileNotFoundError(f"No cotton class directories found: {dataset_root}")
    for class_dir in class_dirs:
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                records.append({
                    "class_label": class_dir.name,
                    "source_group_id": original_group_id(class_dir.name, path),
                    "source_path": path,
                })
    if not records:
        raise FileNotFoundError(f"No cotton images found: {dataset_root}")
    return records


def assign_group_splits(records, seed):
    groups_by_class = defaultdict(set)
    for record in records:
        groups_by_class[record["class_label"]].add(record["source_group_id"])
    assignments = {}
    for class_index, (class_label, groups) in enumerate(sorted(groups_by_class.items())):
        ordered = sorted(groups)
        random.Random(seed + class_index * 1009).shuffle(ordered)
        count = len(ordered)
        train_end = max(1, round(count * 0.70))
        val_end = min(count - 1, train_end + max(1, round(count * 0.15)))
        for index, group in enumerate(ordered):
            assignments[group] = (
                "train" if index < train_end else "val" if index < val_end else "test"
            )
    return assignments


def select_balanced(records, total, seed):
    classes = sorted(set(record["class_label"] for record in records))
    base, remainder = divmod(total, len(classes))
    chosen = []
    for class_index, class_label in enumerate(classes):
        target = base + (class_index < remainder)
        split_targets = {
            "train": round(target * 0.70),
            "val": round(target * 0.15),
        }
        split_targets["test"] = target - split_targets["train"] - split_targets["val"]
        by_split = defaultdict(list)
        for record in records:
            if record["class_label"] == class_label:
                by_split[record["split"]].append(record)
        rng = random.Random(seed + class_index * 2017)
        for split in SPLITS:
            # Prefer one representative per original-photo group before using
            # additional augmented derivatives from a selected group.
            grouped = defaultdict(list)
            for record in by_split[split]:
                grouped[record["source_group_id"]].append(record)
            group_names = sorted(grouped)
            rng.shuffle(group_names)
            ordered = []
            for group in group_names:
                rng.shuffle(grouped[group])
                ordered.append(grouped[group][0])
            for group in group_names:
                ordered.extend(grouped[group][1:])
            needed = split_targets[split]
            if len(ordered) < needed:
                raise RuntimeError(
                    f"Not enough class={class_label} split={split} images: "
                    f"need {needed}, found {len(ordered)}"
                )
            chosen.extend(ordered[:needed])
    return chosen


def _sample_id(record):
    source = Path(record["source_path"])
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:8]
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", source.stem).strip("_")
    return f"cotton_{record['class_label']}_{safe_stem}_{digest}"


def prepare(args):
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output is not empty: {output_root}. Use --overwrite to replace it."
        )
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    records = collect_records(dataset_root)
    assignments = assign_group_splits(records, args.seed)
    for record in records:
        record["split"] = assignments[record["source_group_id"]]
    selected = select_balanced(records, args.total, args.seed)

    manifest_rows = []
    for record in selected:
        split = record["split"]
        identifier = _sample_id(record)
        image_dir = output_root / split / "images"
        mask_dir = output_root / split / "masks"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        source_path = Path(record["source_path"])
        image_path = image_dir / f"{identifier}{source_path.suffix.lower()}"
        mask_path = mask_dir / f"{identifier}.png"
        shutil.copy2(source_path, image_path)
        with Image.open(source_path) as image:
            zero_mask(image.size).save(mask_path)
        manifest_rows.append({
            "sample_id": identifier,
            "surface": "cotton",
            "class_label": record["class_label"],
            # Retained for compatibility with the segmentation manifest and
            # review gallery. Every cotton image requires mask review.
            "is_cracked": 1,
            "source_group_id": record["source_group_id"],
            "split": split,
            "image_path": image_path.relative_to(output_root),
            "mask_path": mask_path.relative_to(output_root),
            "mask_status": "draft_needs_review",
            "source_path": source_path,
        })

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    counts = Counter((row["split"], row["class_label"]) for row in manifest_rows)
    print(f"Prepared {len(manifest_rows)} cotton mask templates at {output_root}")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")
    print(f"Manifest: {manifest_path}")
    print("All masks are blank drafts. Paint lesion pixels white, leave background black,")
    print("then set mask_status=approved_manual after reviewing each image.")
    return manifest_path


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=root / "data" / "augmented_cotton_dataset_v2",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "cotton_workspace_600",
    )
    parser.add_argument("--total", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.total < 40:
        parser.error("--total must be at least 40")
    return args


if __name__ == "__main__":
    prepare(parse_args())
