# SDNET2018 segmentation pilot

This folder prepares a balanced annotation pilot from SDNET2018 without mixing
patches from the same original photograph across train, validation, and test.

The default run selects 300 cracked patches and 300 uncracked patches:

- 100 cracked and 100 uncracked bridge-deck patches
- 100 cracked and 100 uncracked pavement patches
- 100 cracked and 100 uncracked wall patches
- approximately 70% train, 15% validation, and 15% test

Uncracked images receive valid all-zero masks from their dataset label. Cracked
images receive algorithmic draft masks to reduce annotation effort. Draft masks
are not ground truth and must be reviewed and corrected before model training.

## Prepare the pilot

From this directory, run:

```bash
python prepare_pilot.py
python validate_pilot.py
python build_review_gallery.py
```

The default source is:

```text
~/Downloads/DATA_Maguire_20180517_ALL/SDNET2018
```

The generated workspace contains:

```text
workspace/
  manifest.csv
  train/images and train/masks
  val/images and val/masks
  test/images and test/masks
  review/index.html
  review/overlays
```

Open `workspace/review/index.html` in a browser. Red pixels show the current
foreground mask. Correct each cracked mask with an image-annotation tool, export
the result as an 8-bit PNG with values 0 and 255, and update `mask_status` in the
manifest from `draft_needs_review` to `approved_manual`.

Run `validate_pilot.py` after editing. It checks pair existence, dimensions,
binary mask values, all-zero uncracked masks, and original-photo split leakage.

Choose a total cracked count between 100 and 2,000. The
script selects the same number of uncracked samples:

```bash
python prepare_pilot.py --cracked-total 150
```

To build the separate 2,000-image expansion while retaining matching masks
already approved in the original pilot:

```bash
python prepare_pilot.py --cracked-total 1000 \
  --output-root workspace_2000 \
  --approved-workspace workspace
python validate_pilot.py --workspace workspace_2000
python build_review_gallery.py --workspace workspace_2000
```

The corresponding 4,000-image expansion uses 2,000 cracked and 2,000
uncracked images and can carry forward approvals from the 2,000-image set:

```bash
python prepare_pilot.py --cracked-total 2000 \
  --output-root workspace_4000 \
  --approved-workspace workspace_2000
```

Use `--overwrite` only when intentionally replacing an existing workspace.
