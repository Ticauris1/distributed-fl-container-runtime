"""Prepare a group-safe SDNET2018 crack-segmentation annotation pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from mask_utils import draft_crack_mask, zero_mask


SURFACES = {
    "D": ("deck", "CD", "UD"),
    "P": ("pavement", "CP", "UP"),
    "W": ("wall", "CW", "UW"),
}
SPLITS = ("train", "val", "test")
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


def source_group_id(surface_code: str, path: Path) -> str:
    """Return the original-photograph group encoded before the final dash."""
    parent_id = path.stem.rsplit("-", 1)[0]
    return f"{surface_code}_{parent_id}"


def balanced_targets(total: int) -> dict[str, int]:
    base, remainder = divmod(total, len(SURFACES))
    return {
        code: base + (index < remainder)
        for index, code in enumerate(SURFACES)
    }


def collect_records(dataset_root: Path) -> list[dict[str, object]]:
    records = []
    for surface_code, (surface, cracked_dir, uncracked_dir) in SURFACES.items():
        for is_cracked, class_dir in ((True, cracked_dir), (False, uncracked_dir)):
            directory = dataset_root / surface_code / class_dir
            if not directory.is_dir():
                raise FileNotFoundError(f"Missing SDNET2018 directory: {directory}")
            for path in sorted(directory.glob("*.jpg")):
                records.append(
                    {
                        "surface_code": surface_code,
                        "surface": surface,
                        "class_label": "cracked" if is_cracked else "uncracked",
                        "is_cracked": is_cracked,
                        "source_group_id": source_group_id(surface_code, path),
                        "source_path": path,
                    }
                )
    return records


def assign_group_splits(records: list[dict[str, object]], seed: int) -> dict[str, str]:
    """Assign whole original-photo groups to splits, stratified by surface."""
    groups_by_surface: dict[str, set[str]] = defaultdict(set)
    for record in records:
        groups_by_surface[str(record["surface_code"])].add(
            str(record["source_group_id"])
        )
    assignments = {}
    for surface_code, groups in groups_by_surface.items():
        ordered = sorted(groups)
        random.Random(seed + ord(surface_code)).shuffle(ordered)
        count = len(ordered)
        train_end = max(1, round(count * 0.70))
        val_end = min(count - 1, train_end + max(1, round(count * 0.15)))
        for index, group in enumerate(ordered):
            split = "train" if index < train_end else "val" if index < val_end else "test"
            assignments[group] = split
    return assignments


def choose_balanced_records(
    records: list[dict[str, object]], cracked_total: int, seed: int
) -> list[dict[str, object]]:
    """Select equal cracked/uncracked counts balanced across surfaces and splits."""
    targets = balanced_targets(cracked_total)
    chosen = []
    for surface_code, per_class_target in targets.items():
        for is_cracked in (True, False):
            candidates = [
                record
                for record in records
                if record["surface_code"] == surface_code
                and record["is_cracked"] is is_cracked
            ]
            rng = random.Random(seed + ord(surface_code) + (1000 if is_cracked else 2000))
            # Round-robin across splits to retain representation in each subset.
            by_split = defaultdict(list)
            for record in candidates:
                by_split[str(record["split"])].append(record)
            for values in by_split.values():
                rng.shuffle(values)
            desired = {
                "train": round(per_class_target * 0.70),
                "val": round(per_class_target * 0.15),
            }
            desired["test"] = per_class_target - desired["train"] - desired["val"]
            for split in SPLITS:
                available = by_split[split]
                count = desired[split]
                if len(available) < count:
                    raise RuntimeError(
                        f"Not enough {surface_code}/{is_cracked}/{split} records: "
                        f"need {count}, found {len(available)}"
                    )
                chosen.extend(available[:count])
    return chosen


def sample_id(record: dict[str, object]) -> str:
    source = Path(record["source_path"])
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:8]
    return f"{record['surface_code']}_{record['class_label']}_{source.stem}_{digest}"


def prepare(args: argparse.Namespace) -> Path:
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output is not empty: {output_root}. Use --overwrite to replace it."
        )
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    approved_masks = {}
    if args.approved_workspace is not None:
        approved_workspace = args.approved_workspace.expanduser().resolve()
        approved_manifest = approved_workspace / "manifest.csv"
        with approved_manifest.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["mask_status"] == "approved_manual":
                    approved_masks[row["sample_id"]] = approved_workspace / row["mask_path"]

    records = collect_records(dataset_root)
    assignments = assign_group_splits(records, args.seed)
    for record in records:
        record["split"] = assignments[str(record["source_group_id"])]
    selected = choose_balanced_records(records, args.cracked_total, args.seed)

    manifest_rows = []
    for index, record in enumerate(selected, start=1):
        split = str(record["split"])
        identifier = sample_id(record)
        image_dir = output_root / split / "images"
        mask_dir = output_root / split / "masks"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{identifier}.jpg"
        mask_path = mask_dir / f"{identifier}.png"
        source_path = Path(record["source_path"])
        shutil.copy2(source_path, image_path)
        with Image.open(source_path) as image:
            if bool(record["is_cracked"]):
                prior_mask = approved_masks.get(identifier)
                if prior_mask is not None:
                    with Image.open(prior_mask) as approved:
                        mask = approved.convert("L").copy()
                    if mask.size != image.size:
                        raise ValueError(f"Approved mask size mismatch: {identifier}")
                    status = "approved_manual"
                else:
                    mask = draft_crack_mask(image)
                    status = "draft_needs_review"
            else:
                mask = zero_mask(image.size)
                status = "approved_zero_from_uncracked_label"
            mask.save(mask_path)
        manifest_rows.append(
            {
                "sample_id": identifier,
                "surface": record["surface"],
                "class_label": record["class_label"],
                "is_cracked": int(bool(record["is_cracked"])),
                "source_group_id": record["source_group_id"],
                "split": split,
                "image_path": image_path.relative_to(output_root),
                "mask_path": mask_path.relative_to(output_root),
                "mask_status": status,
                "source_path": source_path,
            }
        )

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    counts = Counter((row["split"], row["surface"], row["class_label"]) for row in manifest_rows)
    print(f"Prepared {len(manifest_rows)} samples at {output_root}")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")
    print(f"Manifest: {manifest_path}")
    print("Cracked masks are drafts and require human review.")
    return manifest_path


def parse_args() -> argparse.Namespace:
    default_dataset = Path.home() / "Downloads" / "DATA_Maguire_20180517_ALL" / "SDNET2018"
    default_output = Path(__file__).resolve().parent / "workspace"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset)
    parser.add_argument("--output-root", type=Path, default=default_output)
    parser.add_argument(
        "--cracked-total",
        type=int,
        default=300,
        help="Total cracked samples; the same number of uncracked samples is selected.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--approved-workspace",
        type=Path,
        help="Carry forward matching approved_manual masks from an earlier workspace.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 100 <= args.cracked_total <= 2000:
        parser.error("--cracked-total must be between 100 and 2000")
    return args


if __name__ == "__main__":
    prepare(parse_args())
