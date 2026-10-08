"""Filesystem discovery helpers for source images."""

import os

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def find_image_files(root_directory):
    """Scan a directory and return image paths paired with directory labels."""
    image_paths, labels = [], []
    if not os.path.isdir(root_directory):
        print(f"❌ ERROR: Directory not found at {root_directory}")
        return image_paths, labels
    for dirpath, _, filenames in os.walk(root_directory):
        if os.path.basename(dirpath).startswith("."):
            continue
        for filename in filenames:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(dirpath, filename))
                labels.append(os.path.basename(dirpath))
    print(
        f"[{os.path.basename(root_directory)}] Found {len(image_paths)} images "
        f"across {len(set(labels))} classes."
    )
    return image_paths, labels


def check_image_paths(paths, label="images"):
    """Raise when any supplied image path does not exist."""
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} missing {label} path(s), e.g. {missing[:5]}"
        )
    return True
