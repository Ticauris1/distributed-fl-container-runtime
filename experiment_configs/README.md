# Experiment configuration

Use `experiment_setup.py` as the control panel for routine experiment changes.
It selects the workflow, seeds, training settings, and enabled run names.

For centralized segmentation, edit `CENTRALIZED_SEGMENTATION_RUNS` in
`experiment_setup.py`. Every enabled
encoder runs with seeds 42, 52, and 62 at 384×384.

For federated segmentation, edit `FEDERATED_SEGMENTATION_RUNS` in
`experiment_setup.py`. The full map contains sample-size
and paper-CELM comparisons for IID and crack-area skew with two and three
clients, but only the named subset is launched.

Classification variants live in `classification.py`. Shared paper-CELM
hyperparameters live in `security.py`.
