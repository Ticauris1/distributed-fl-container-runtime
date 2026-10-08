# Distributed Container Runtime for the Federated Learning Project

This project supports two execution modes:

1. `main.py` runs the complete research pipeline in one Python process.
2. Docker Compose runs a federated server and multiple client workers in separate containers.

The refactored codebase keeps data preparation, models, training, privacy, hybrid inference, reporting, traditional ML, and experiment orchestration in focused packages. The distributed runtime imports the same model factory, dataset, local-training, and aggregation implementations.

## Public Release Scope

This repository contains source code and experiment definitions only. It does
not include research datasets, generated manifests, model checkpoints, runtime
state, archived results, Tableau extracts, or machine-specific paths. Obtain
each dataset from its original provider and review its license before use.

Local dataset locations can be supplied through environment variables:

```bash
export GEOM_ROOT_2=/path/to/combined_images
export CCSS_DATASET_ROOT=/path/to/CCSS-DATA-V4-withVal
export TABLEAU_WORKBOOK_DIR=/path/to/tableau/workbooks
```

The privacy modules are research prototypes. Modular masking and CKKS are
distinct experimental aggregation paths and should not be interpreted as a
production security guarantee without an explicit threat model, key-management
design, authentication, transport security, and independent review.

## Project Structure

```text
project_root/
├── main.py
├── config.py
├── data/
├── models/
├── training/
├── hybrid/
├── privacy/
├── reporting/
├── traditional/
├── experiments/
├── containerization/
│   ├── __init__.py
│   ├── Dockerfile
│   ├── docker-compose.generated.yml
│   ├── container_entrypoint.py
│   ├── distributed_server.py
│   ├── distributed_client.py
│   ├── distributed_train_utils.py
│   ├── prepare_distributed_run.py
│   ├── generate_compose.py
│   ├── fl_api.py
│   └── requirements.txt
├── .dockerignore
└── runtime/
```

The old flat modules such as `server.py`, `client.py`, `models.py`, `data_utils.py`, and `plotting.py` were removed after their logic was migrated into the package structure.

All Docker runtime files are grouped under `containerization/`. The `.dockerignore` file intentionally remains at the project root because the Docker build context is the complete project directory.

## Single-Process Research Pipeline

The primary local entry point is:

```bash
python main.py
```

`main.py` now handles orchestration only:

- runtime initialization
- dataset preparation
- experiment-grid construction
- dispatch to deep or traditional runners
- final completion reporting

Experiment behavior is controlled from `config.py`.

## Distributed Docker Architecture

```text
Docker Compose
├── fed-server
│   ├── creates and owns the global model
│   ├── selects clients for each round
│   ├── receives update metadata over HTTP
│   ├── loads client updates from the shared runtime volume
│   └── applies FedAvg or FedAdam aggregation
│
├── fed-client-1
│   ├── reads client_1.csv
│   ├── trains only on client 1 data
│   └── submits its update to the server
│
├── fed-client-2
│   ├── reads client_2.csv
│   ├── trains only on client 2 data
│   └── submits its update to the server
│
└── fed-client-N
    └── generated from runtime/plan.json
```

The generated Compose file mounts:

```text
../runtime  -> /app/runtime
../data     -> /app/data (read-only)
../models   -> /app/models (read-only)
../results  -> role-specific result directories
```

These paths are relative to `containerization/docker-compose.generated.yml` and therefore still select the project-root directories.

Docker Desktop on macOS normally runs this workflow on CPU. Apple MPS is not available inside the Linux containers.

## Prerequisites

- Docker Desktop is installed and running.
- `docker compose version` succeeds.
- The project has a valid `containerization/requirements.txt`.
- The host Python environment used for preparation has the project dependencies installed.

Check Docker:

```bash
docker version
docker compose version
```

Install the Python dependencies in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

Datasets and generated runtime files are intentionally not committed. Configure
the desired dataset in `config.py` or through the environment variables above,
then generate the runtime plan and Compose file.

From the project root:

```bash
cd /path/to/distributed-fl-container-runtime

python -m containerization.prepare_distributed_run
python -m containerization.generate_compose
docker compose -f containerization/docker-compose.generated.yml down --remove-orphans
docker compose -f containerization/docker-compose.generated.yml up --build
```

To run in the background:

```bash
docker compose -f containerization/docker-compose.generated.yml up --build -d
docker compose -f containerization/docker-compose.generated.yml logs -f
```

To stop the runtime:

```bash
docker compose -f containerization/docker-compose.generated.yml down --remove-orphans
```

## Regenerating the Runtime Plan

Regenerate the plan when changing the dataset, model, client count, partition distribution, or training configuration.

### Important Dataset Setting

The generated Compose file mounts only the project-local `data/` directory. The current `Geometric V2` path in `config.py` points to an external host directory:

```text
/path/to/combined_images
```

That host path is not visible inside the containers. Before regenerating the Docker plan, the recommended setting is:

```python
RUN_ON_DATASET = "Geometric"
```

The `Geometric` dataset is stored under the mounted project directory:

```text
data/augmented_cotton_dataset_v2
```

After updating `config.py`, run:

```bash
cd /path/to/distributed-fl-container-runtime

python -m containerization.prepare_distributed_run
python -m containerization.generate_compose
docker compose -f containerization/docker-compose.generated.yml down --remove-orphans
docker compose -f containerization/docker-compose.generated.yml up --build
```

Preparation creates:

```text
runtime/plan.json
runtime/manifests/client_1.csv
runtime/manifests/client_2.csv
runtime/manifests/server_val.csv
runtime/manifests/test.csv
```

Compose generation reads `runtime/plan.json` and creates the correct number of `fed-client-N` services in `containerization/docker-compose.generated.yml`.

## Runtime Overrides

Some distributed settings can be overridden while generating the plan:

```bash
DISTRIBUTED_NUM_CLIENTS=2 \
DISTRIBUTED_MODEL=LW-efficientnet_b0 \
SERVER_METHOD=FedAvg \
python -m containerization.prepare_distributed_run

python -m containerization.generate_compose
```

Available overrides include:

- `DISTRIBUTED_NUM_CLIENTS`
- `DISTRIBUTED_MODEL`
- `DISTRIBUTED_DIST`
- `SERVER_METHOD`
- `RUNTIME_DIR`

Other settings—rounds, local epochs, learning rate, clients per round, and batch size—come from `config.py`.

## Verifying Manifest Paths

Docker manifests must contain container-visible paths:

```bash
head runtime/manifests/client_1.csv
```

Correct:

```text
image_path,label
/app/data/augmented_cotton_dataset_v2/0/example.jpg,0
```

Incorrect:

```text
/path/on/the/host/...
```

If a manifest contains a host-only absolute path, the selected dataset is outside the directory mounted at `/app/data`. Use a project-local dataset or extend both the Compose mounts and manifest path conversion for the external dataset.

## How the Distributed Runtime Works

### Preparation

`containerization/prepare_distributed_run.py`:

- selects the configured dataset
- discovers and encodes image labels
- creates train, server-validation, and test splits
- partitions training indices across clients
- writes client manifests
- writes `runtime/plan.json`

### Server

`containerization/distributed_server.py`:

- imports `models.factory.make_dl_model`
- imports aggregation from `training.aggregation`
- creates the global model
- exposes assignment and update endpoints through FastAPI
- selects clients deterministically by round
- aggregates submitted state dictionaries
- writes global checkpoints to `runtime/checkpoints/`

### Clients

`containerization/distributed_client.py` and `containerization/distributed_train_utils.py`:

- request assignments from the server
- load the current global checkpoint
- load only the assigned client manifest
- train with `training.local.epoch_train`
- save an update under `runtime/updates/`
- notify the server through HTTP

## Monitoring

Follow all services:

```bash
docker compose -f containerization/docker-compose.generated.yml logs -f
```

Follow one service:

```bash
docker logs -f fed-server
docker logs -f fed-client-1
docker logs -f fed-client-2
```

Check the server API:

```bash
curl http://localhost:8000/health
```

Inspect running containers:

```bash
docker compose -f containerization/docker-compose.generated.yml ps
```

## Smoke-Test Settings

Deep-learning clients can consume substantial memory when multiple CPU containers train simultaneously. For an initial Docker test, use:

```text
FED_ROUNDS = 2
FED_LOCAL_EPOCHS = 1
BATCH_SIZE = 4
FED_CLIENTS_PER_ROUND = 1
```

Set these values in `config.py`, regenerate the plan and Compose file, and then start the containers.

If Docker kills a client, inspect it with:

```bash
docker inspect fed-client-2 \
  --format 'ExitCode={{.State.ExitCode}} OOMKilled={{.State.OOMKilled}} Error={{.State.Error}}'
```

Docker Desktop resource guidance for this workload:

```text
Memory: 8 GB minimum when available
Swap: 2–4 GB
```

## Rebuilding After Code Changes

Code is copied into the image during `docker build`. After changing Python files, rebuild:

```bash
docker compose -f containerization/docker-compose.generated.yml down --remove-orphans
docker compose -f containerization/docker-compose.generated.yml up --build
```

To force a completely fresh image build:

```bash
docker compose -f containerization/docker-compose.generated.yml build --no-cache
docker compose -f containerization/docker-compose.generated.yml up
```

## Common Commands

```bash
# Start in the foreground
docker compose -f containerization/docker-compose.generated.yml up --build

# Start in the background
docker compose -f containerization/docker-compose.generated.yml up --build -d

# Follow logs
docker compose -f containerization/docker-compose.generated.yml logs -f

# Show container status
docker compose -f containerization/docker-compose.generated.yml ps

# Stop and remove services
docker compose -f containerization/docker-compose.generated.yml down --remove-orphans

# Remove conflicting named containers
docker rm -f fed-server fed-client-1 fed-client-2
```

## Current Distributed-Runtime Limitations

- Final evaluation plots and the full single-process summary workflow are not yet reproduced by the distributed server.
- Hybrid inference and Bayesian threshold optimization are available in the single-process runner but not the distributed server.
- The privacy-aware aggregation package is used by single-process federated training but is not yet applied by `containerization/distributed_server.py`.
- Setting the distributed plan method to `FedProx` does not yet add the proximal term to client-container training; true FedProx is currently implemented in `training/fedprox.py` for the single-process runner.
- Checkpoint exchange uses the shared `runtime/` volume; HTTP currently coordinates metadata rather than transferring model binaries.
- Docker Desktop on macOS uses CPU and may require smoke-test settings.

## Troubleshooting

### `runtime/plan.json` is missing

```bash
python -m containerization.prepare_distributed_run
python -m containerization.generate_compose
```

### A dataset directory cannot be found

Confirm `RUN_ON_DATASET` and `DATASETS` in `config.py`. For the generated Docker mounts, use the project-local `Geometric` dataset.

### Containers have stale code

```bash
docker compose -f containerization/docker-compose.generated.yml up --build
```

Use `build --no-cache` if Docker still reuses an old layer.

### A container name already exists

```bash
docker compose -f containerization/docker-compose.generated.yml down --remove-orphans
docker rm -f fed-server fed-client-1 fed-client-2
```

### A client was killed for memory use

Reduce rounds, local epochs, batch size, and clients per round. Then regenerate the runtime plan.

## Research and Documentation Value

The project maintains a reproducible single-process research pipeline and a containerized distributed runtime. In the distributed version, the server and clients run as separate containers, each client trains only on its assigned manifest, communication is coordinated through explicit HTTP endpoints, and aggregation is performed by the server container. This more closely models decentralized federated learning and provides a foundation for authentication, TLS, signed updates, audit logging, and privacy-aware distributed aggregation.
