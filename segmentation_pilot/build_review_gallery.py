"""Build an offline HTML gallery for reviewing draft segmentation masks."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

from PIL import Image

from mask_utils import overlay_mask


def build_gallery(workspace: Path, limit: int | None = None) -> Path:
    workspace = workspace.expanduser().resolve()
    manifest = workspace / "manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["is_cracked"] == "1"]
    if limit is not None:
        rows = rows[:limit]
    preview_dir = workspace / "review" / "overlays"
    preview_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for row in rows:
        image_path = workspace / row["image_path"]
        mask_path = workspace / row["mask_path"]
        preview_path = preview_dir / f"{row['sample_id']}.jpg"
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            overlay_mask(image, mask).save(preview_path, quality=90)
        cards.append(
            "<article>"
            f"<h3>{html.escape(row['sample_id'])}</h3>"
            f"<p>{html.escape(row['surface'])} | {html.escape(row['split'])} | "
            f"{html.escape(row['mask_status'])}</p>"
            f"<img src='{html.escape(preview_path.relative_to(workspace / 'review').as_posix())}' "
            "loading='lazy'>"
            f"<code>{html.escape(row['mask_path'])}</code>"
            "</article>"
        )
    is_cotton = bool(rows) and all(row.get("surface") == "cotton" for row in rows)
    title = "Cotton lesion-mask review" if is_cotton else "SDNET2018 crack-mask review"
    foreground = "lesion/disease regions" if is_cotton else "crack regions"
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body{{font-family:system-ui;margin:24px;background:#f4f5f7;color:#17202a}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:18px}}
article{{background:white;padding:14px;border-radius:10px;box-shadow:0 1px 5px #bbb}}
img{{width:100%;image-rendering:auto}} h3{{font-size:14px;overflow-wrap:anywhere}}
code{{font-size:11px;overflow-wrap:anywhere}} p{{font-size:12px}}
</style></head><body><h1>{title}</h1>
<p>Red pixels are annotated {foreground}. Edit the PNG mask and change its manifest
status to <code>approved_manual</code> only after review.</p><main>{''.join(cards)}</main></body></html>"""
    output = workspace / "review" / "index.html"
    output.write_text(document, encoding="utf-8")
    print(f"Review gallery: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace", type=Path, default=Path(__file__).resolve().parent / "workspace"
    )
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()
    build_gallery(arguments.workspace, arguments.limit)
