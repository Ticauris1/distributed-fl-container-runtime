# Publication Review Checklist

This directory is a sanitized review copy. It has not been published and is
not connected to a GitHub repository.

## Excluded from the review copy

- Raw classification and segmentation datasets
- Generated images, masks, and dataset manifests
- Model checkpoints and serialized weights
- Experiment results and archived results
- Runtime plans, client manifests, updates, and checkpoints
- Tableau workbooks, extracts, and temporary files
- Python caches, local environments, logs, and operating-system metadata
- Machine-specific backup configuration files

## Portability changes

- Personal absolute paths were replaced with environment variables or
  `/path/to/...` examples.
- `.env.example` documents optional local path variables.
- `.gitignore` prevents datasets, manifests, results, checkpoints, and local
  configuration from being committed later.
- The public README explains that datasets and experimental artifacts are not
  distributed with the source.

## Checks completed

- No files larger than 5 MB remain.
- No model checkpoint or Tableau extract formats remain.
- No personal home-directory paths remain.
- No common GitHub tokens, cloud access keys, private keys, or populated
  password/token assignments were detected by the local pattern scan.
- Python source compilation completed successfully.

Before publication, review the README, active experiment defaults, dependency
versions, third-party dataset citations, and the MIT license attribution.
