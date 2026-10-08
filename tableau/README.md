# Tableau reporting layer

The Python experiment figures remain the source for prediction masks and
automated diagnostics. Tableau consumes the consolidated CSV files in
`results/tableau/`; TabPy supplies calculations that are awkward to maintain as
Tableau formulas.

## Refresh the datasets

```bash
python -m reporting.tableau_export
```

The export scans both active and archived results. Use `result_status` as a
Tableau filter to switch between final and exploratory work.

## Recommended dashboards

### Research overview

- KPI tiles: mean Dice, IoU, precision, recall, and inference time.
- Encoder comparison: horizontal dot plot with Dice and seed-level points.
- Centralized versus federated: paired slope chart.
- Filters: model, encoder, dataset, seed, result status, and resolution.

### Federated learning

- Validation Dice and loss by communication round.
- IID versus crack-area skew grouped bars.
- Sample-size versus paper-CELM paired seed plot.
- Client aggregation weight by round and client.
- Filters: client count, distribution, weighting, method, and seed.

### Training diagnostics

- Centralized train/validation loss by epoch.
- Validation Dice and IoU by epoch.
- Best epoch/round annotation.
- Small multiples by encoder.

### Efficiency and security

- Dice versus milliseconds per image scatter plot.
- Parameter count and checkpoint size.
- CELM estimation time by seed and distribution.
- CKKS encryption, aggregation, decryption time, ciphertext bytes, and error
  when those experiments are enabled.

Use a restrained palette: navy for centralized, teal for sample-size FedAvg,
amber for paper-CELM, and muted red only for failures or regressions. Keep seed
points visible behind mean markers rather than showing means alone.

## TabPy

Create a separate environment and start TabPy locally:

```bash
python -m venv .tabpy-env
source .tabpy-env/bin/activate
pip install -r tableau/requirements.txt
tabpy
```

In another terminal:

```bash
source .tabpy-env/bin/activate
python -m tableau.deploy_tabpy_models --endpoint http://127.0.0.1:9004/
```

Connect Tableau Desktop to `localhost:9004` as an Analytics Extension. Keep
the server bound to localhost for individual research use. Configure
authentication and TLS before allowing remote access.

Named functions deployed by the project:

- `fl_mean`
- `fl_confidence_interval_95`
- `fl_paired_delta`
- `fl_rolling_mean`
- `fl_pareto_score`

Tableau Desktop is required to create and visually verify the final workbook.
It was not installed when this reporting layer was generated.
