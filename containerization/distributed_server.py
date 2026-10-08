"""
Distributed federated server.

This runs inside the fed-server container.

It uses your research logic for:
- model construction through models.factory.make_dl_model
- aggregation through training.aggregation.weighted_average_weights
- optional FedAdam through training.aggregation.FedAdamServer

The server coordinates rounds over HTTP. Clients train locally in separate
containers and submit state_dict files back to the shared runtime volume.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import copy
import json
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException

import config
from .fl_api import RoundAssignment, ClientUpdate
from models.factory import make_dl_model
from training.aggregation import weighted_average_weights
from training.celm import CELMContributionEstimator, PaperCELMContributionEstimator

try:
    from training.aggregation import FedAdamServer
except Exception:
    FedAdamServer = None

app = FastAPI(title="Distributed Federated Server")

RUNTIME_DIR = Path(os.getenv("RUNTIME_DIR", "/app/runtime"))
PLAN_PATH = RUNTIME_DIR / "plan.json"
PLAN = json.loads(PLAN_PATH.read_text()) if PLAN_PATH.exists() else {}
DATASET_CHECKPOINT_DIR = re.sub(
    r"[^a-z0-9]+", "_", str(PLAN.get("source_name", "unknown_dataset")).lower()
).strip("_")
CHECKPOINT_DIR = RUNTIME_DIR / "checkpoints" / DATASET_CHECKPOINT_DIR
UPDATES_DIR = RUNTIME_DIR / "updates"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
UPDATES_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_ROUND = 0
SELECTED_BY_ROUND: Dict[int, List[int]] = {}
UPDATES_BY_ROUND: Dict[int, Dict[int, ClientUpdate]] = {}
GLOBAL_MODEL = None
FEDADAM_SERVER = None
TRAINING_DONE = False
CELM_ESTIMATOR = None


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def initialize_if_needed():
    global GLOBAL_MODEL, FEDADAM_SERVER, CELM_ESTIMATOR

    if GLOBAL_MODEL is not None:
        return

    if not PLAN:
        raise RuntimeError(f"Missing runtime plan at {PLAN_PATH}")

    config.set_seed(config.RANDOM_STATE)
    device = get_device()
    GLOBAL_MODEL = make_dl_model(
        model_name=PLAN["model_name"],
        num_classes=int(PLAN["num_classes"]),
        device=device,
        pretrained=not bool(getattr(config, "FED_COLD_START", False)),
    )

    method = PLAN.get("server_method", "FedAvg").lower()
    if method == "fedadam":
        if FedAdamServer is None:
            raise RuntimeError("FedAdamServer could not be imported from training.aggregation")
        FEDADAM_SERVER = FedAdamServer(
            model=GLOBAL_MODEL,
            lr=float(os.getenv("FEDADAM_SERVER_LR", "0.001")),
            beta1=float(os.getenv("FEDADAM_BETA1", "0.9")),
            beta2=float(os.getenv("FEDADAM_BETA2", "0.999")),
            eps=float(os.getenv("FEDADAM_EPS", "1e-8")),
        )

    weight_method = PLAN.get("client_weight_method", "sample_size").lower()
    if weight_method not in {"sample_size", "uniform", "celm", "paper_celm"}:
        raise ValueError(f"Unsupported client_weight_method={weight_method}")
    if weight_method == "celm":
        if method == "fedprox":
            raise ValueError("CELM container weighting is not implemented for FedProx")
        CELM_ESTIMATOR = CELMContributionEstimator(
            PLAN["model_name"], int(PLAN["num_classes"]), int(PLAN["n_clients"]),
            int(PLAN["rounds"]), device,
            warmup_rounds=int(PLAN["celm_warmup_rounds"]),
            warmup_fraction=float(PLAN["celm_warmup_fraction"]),
            steps=int(PLAN["celm_steps"]), lr=float(PLAN["celm_lr"]),
            l2_penalty=float(PLAN["celm_l2_penalty"]),
            ema_beta=float(PLAN["celm_ema_beta"]),
            seed=int(PLAN.get("random_state", 42)),
        )
    elif weight_method == "paper_celm":
        if int(PLAN["clients_per_round"]) != int(PLAN["n_clients"]):
            raise ValueError("paper_celm requires full client participation")
        if method == "fedprox":
            raise ValueError("paper_celm container weighting is not implemented for FedProx")
        CELM_ESTIMATOR = PaperCELMContributionEstimator(
            PLAN["model_name"], int(PLAN["num_classes"]), int(PLAN["n_clients"]),
            int(PLAN["rounds"]), device,
            warmup_rounds=int(PLAN["paper_celm_warmup_rounds"]),
            warmup_fraction=float(PLAN["paper_celm_warmup_fraction"]),
            steps=int(PLAN["paper_celm_steps"]),
            lr=float(PLAN["paper_celm_lr"]),
            l2_penalty=float(PLAN["paper_celm_l2_penalty"]),
            ema_beta=float(PLAN["paper_celm_ema_beta"]),
            seed=int(PLAN.get("random_state", 42)),
        )

    save_global_state(0)


def save_global_state(round_id: int) -> str:
    path = CHECKPOINT_DIR / f"global_round_{round_id}.pt"
    torch.save(GLOBAL_MODEL.state_dict(), path)
    return str(path)


def select_clients(round_id: int):
    n_clients = int(PLAN["n_clients"])
    cpr = min(int(PLAN["clients_per_round"]), n_clients)
    rng = random.Random(int(getattr(config, "RANDOM_STATE", 42)) + round_id)
    selected = sorted(rng.sample(list(range(1, n_clients + 1)), cpr))
    SELECTED_BY_ROUND[round_id] = selected
    return selected


def aggregate_round(round_id: int):
    global GLOBAL_MODEL

    updates = UPDATES_BY_ROUND.get(round_id, {})
    selected = SELECTED_BY_ROUND.get(round_id, [])

    if len(updates) < len(selected):
        return {
            "status": "waiting",
            "round_id": round_id,
            "received": len(updates),
            "expected": len(selected),
        }

    device = get_device()
    current_state = copy.deepcopy(GLOBAL_MODEL.state_dict())

    local_weights = []
    datalens = []

    for cid in selected:
        upd = updates[cid]
        state = torch.load(upd.update_path, map_location=device)
        local_weights.append(state)
        datalens.append(int(upd.num_samples))

    weight_method = PLAN.get("client_weight_method", "sample_size").lower()
    if weight_method in {"celm", "paper_celm"}:
        if CELM_ESTIMATOR is None:
            raise RuntimeError("CELM was requested but its estimator is not initialized")
        aggregation_weights, weighting_stats = CELM_ESTIMATOR.weights_for_round(
            round_id - 1, current_state, local_weights,
            [client_id - 1 for client_id in selected],
        )
    elif weight_method == "sample_size":
        aggregation_weights = np.asarray(datalens, dtype=np.float64)
        phase = "sample_size"
        normalized_weights = aggregation_weights / aggregation_weights.sum()
        weighting_stats = {
            "status": "success", "fallback_used": False, "round": round_id,
            "phase": phase, "method": weight_method,
            "selected_client_ids": selected,
            "client_weights": normalized_weights.tolist(),
            "estimation_seconds": 0.0,
        }
    else:
        aggregation_weights = np.ones(len(local_weights), dtype=np.float64)
        phase = "uniform"
        normalized_weights = aggregation_weights / aggregation_weights.sum()
        weighting_stats = {
            "status": "success", "fallback_used": False, "round": round_id,
            "phase": phase, "method": weight_method,
            "selected_client_ids": selected,
            "client_weights": normalized_weights.tolist(),
            "estimation_seconds": 0.0,
        }
    weighting_stats["method"] = weight_method
    with (RUNTIME_DIR / "client_weighting_rounds.jsonl").open("a") as stream:
        stream.write(json.dumps(weighting_stats) + "\n")
    averaged_state = weighted_average_weights(local_weights, aggregation_weights)
    method = PLAN.get("server_method", "FedAvg").lower()

    if method == "fedavg" or method == "fedprox":
        GLOBAL_MODEL.load_state_dict(averaged_state, strict=True)

    elif method == "fedadam":
        if FEDADAM_SERVER is None:
            raise RuntimeError("FedAdam was requested but FEDADAM_SERVER is not initialized")

        deltas = []
        for name, _ in GLOBAL_MODEL.named_parameters():
            deltas.append(
                averaged_state[name].to(device) - current_state[name].to(device)
            )
        FEDADAM_SERVER.step(deltas)

        # Copy non-parameter buffers from the averaged state.
        with torch.no_grad():
            new_state = GLOBAL_MODEL.state_dict()
            param_names = {name for name, _ in GLOBAL_MODEL.named_parameters()}
            for key in new_state:
                if key not in param_names and key in averaged_state:
                    new_state[key].copy_(averaged_state[key].to(device))

    else:
        raise ValueError(f"Unsupported server_method={method}")

    global_path = save_global_state(round_id)

    return {
        "status": "aggregated",
        "round_id": round_id,
        "global_state_path": global_path,
        "selected_clients": selected,
        "num_updates": len(local_weights),
        "total_samples": int(sum(datalens)),
        "client_weight_method": weight_method,
        "client_weights": weighting_stats["client_weights"],
        "client_weighting_seconds": weighting_stats["estimation_seconds"],
    }


@app.on_event("startup")
def startup_event():
    initialize_if_needed()
    rounds = int(PLAN["rounds"])
    for r in range(1, rounds + 1):
        select_clients(r)
    print(f"[server] initialized distributed run with plan: {PLAN}", flush=True)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "role": "server",
        "plan_loaded": bool(PLAN),
        "rounds": PLAN.get("rounds"),
        "n_clients": PLAN.get("n_clients"),
    }


@app.get("/assignment/{client_id}/{round_id}", response_model=RoundAssignment)
def assignment(client_id: int, round_id: int):
    initialize_if_needed()

    rounds = int(PLAN["rounds"])
    if round_id > rounds:
        return RoundAssignment(
            status="done",
            round_id=round_id,
            client_id=client_id,
            selected=False,
        )

    selected = SELECTED_BY_ROUND.get(round_id)
    if selected is None:
        selected = select_clients(round_id)

    global_state_path = CHECKPOINT_DIR / f"global_round_{round_id - 1}.pt"
    if not global_state_path.exists():
        # If previous round did not finish yet, client should wait.
        return RoundAssignment(
            status="waiting",
            round_id=round_id,
            client_id=client_id,
            selected=False,
        )

    return RoundAssignment(
        status="ready",
        round_id=round_id,
        client_id=client_id,
        selected=client_id in selected,
        global_state_path=str(global_state_path),
        model_name=PLAN["model_name"],
        source_name=PLAN["source_name"],
        num_classes=int(PLAN["num_classes"]),
        local_epochs=int(PLAN["local_epochs"]),
        lr=float(PLAN["lr"]),
    )


@app.post("/submit_update")
def submit_update(update: ClientUpdate):
    UPDATES_BY_ROUND.setdefault(update.round_id, {})[update.client_id] = update
    result = aggregate_round(update.round_id)
    return {
        "status": "received",
        "client_id": update.client_id,
        "round_id": update.round_id,
        "aggregation": result,
    }


@app.get("/round/{round_id}/status")
def round_status(round_id: int):
    selected = SELECTED_BY_ROUND.get(round_id, [])
    updates = UPDATES_BY_ROUND.get(round_id, {})
    return {
        "round_id": round_id,
        "selected": selected,
        "received_clients": sorted(list(updates.keys())),
        "received": len(updates),
        "expected": len(selected),
        "global_checkpoint_exists": (CHECKPOINT_DIR / f"global_round_{round_id}.pt").exists(),
    }


def run_server_loop():
    """
    Optional blocking loop for logs. FastAPI handles requests.
    """
    initialize_if_needed()
    print("[server] API ready. Waiting for clients.", flush=True)
    while True:
        time.sleep(30)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "containerization.distributed_server:app",
        host=os.getenv("FED_SERVER_HOST", "0.0.0.0"),
        port=int(os.getenv("FED_SERVER_PORT", "8000")),
        reload=False,
    )
