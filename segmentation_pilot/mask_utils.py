"""Draft crack-mask generation using only Pillow and NumPy."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def zero_mask(size: tuple[int, int]) -> Image.Image:
    """Return a binary all-background mask using the requested (width, height)."""
    return Image.new("L", size, color=0)


def draft_crack_mask(image: Image.Image, percentile: float = 94.0) -> Image.Image:
    """Create a conservative dark-line pseudo-mask for manual review.

    SDNET2018 contains shadows, joints, debris, and texture that can be mistaken
    for cracks. This result is deliberately marked as a draft by the preparation
    script and must not be treated as ground truth without human correction.
    """
    gray = image.convert("L")
    # A broad maximum filter estimates the brighter local concrete background.
    local_background = gray.filter(ImageFilter.MaxFilter(15))
    response = np.asarray(local_background, dtype=np.int16) - np.asarray(
        gray, dtype=np.int16
    )
    positive = response[response > 0]
    if positive.size == 0:
        return zero_mask(gray.size)
    threshold = max(12.0, float(np.percentile(positive, percentile)))
    binary = (response >= threshold).astype(np.uint8) * 255
    mask = Image.fromarray(binary, mode="L")
    # Remove isolated pixels, then reconnect narrow crack fragments.
    mask = mask.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    return mask.point(lambda value: 255 if value else 0, mode="L")


def overlay_mask(image: Image.Image, mask: Image.Image, alpha: float = 0.45) -> Image.Image:
    """Overlay mask pixels in red for a review preview."""
    base = image.convert("RGB")
    binary = np.asarray(mask.convert("L")) > 0
    overlay = np.asarray(base, dtype=np.float32).copy()
    overlay[binary] = (
        (1.0 - alpha) * overlay[binary]
        + alpha * np.array([255.0, 0.0, 0.0], dtype=np.float32)
    )
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")
