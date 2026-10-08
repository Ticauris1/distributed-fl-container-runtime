"""
Distributed federated client.

Each client container:
1. Reads CLIENT_ID.
2. Asks the server if it is selected for a round.
3. If selected, loads its manifest and the global checkpoint.
4. Trains locally using training.local.epoch_train through train_one_client_round.
5. Saves its local update to the shared runtime volume.
6. Submits update metadata to the server.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import time

import requests

from .distributed_train_utils import train_one_client_round

CLIENT_ID = int(os.getenv("CLIENT_ID", "1"))
FED_SERVER_URL = os.getenv("FED_SERVER_URL", "http://fed-server:8000")
RUNTIME_DIR = Path(os.getenv("RUNTIME_DIR", "/app/runtime"))
PLAN_PATH = RUNTIME_DIR / "plan.json"
UPDATES_DIR = RUNTIME_DIR / "updates"
UPDATES_DIR.mkdir(parents=True, exist_ok=True)


def wait_for_server(timeout_seconds: int = 120):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            r = requests.get(f"{FED_SERVER_URL}/health", timeout=3)
            if r.ok:
                print(f"[client {CLIENT_ID}] connected: {r.json()}", flush=True)
                return
        except Exception as exc:
            print(f"[client {CLIENT_ID}] waiting for server: {exc}", flush=True)
        time.sleep(2)
    raise RuntimeError(f"Client {CLIENT_ID} could not reach {FED_SERVER_URL}")


def get_assignment(round_id: int):
    while True:
        r = requests.get(
            f"{FED_SERVER_URL}/assignment/{CLIENT_ID}/{round_id}",
            timeout=10,
        )
        r.raise_for_status()
        assignment = r.json()

        if assignment["status"] in {"ready", "done"}:
            return assignment

        print(
            f"[client {CLIENT_ID}] round {round_id} waiting for assignment/checkpoint",
            flush=True,
        )
        time.sleep(3)


def main():
    wait_for_server()

    if not PLAN_PATH.exists():
        raise RuntimeError(f"Missing {PLAN_PATH}")

    plan = json.loads(PLAN_PATH.read_text())
    rounds = int(plan["rounds"])
    batch_size = int(plan["batch_size"])
    weight_method = str(plan.get("client_weight_method", "sample_size")).lower()
    paper_warmup_rounds = int(plan.get("paper_celm_warmup_rounds", 0))
    local_classifier_path = (
        RUNTIME_DIR / "client_state" / f"client_{CLIENT_ID}_classifier.pt"
    )

    manifest_path = RUNTIME_DIR / "manifests" / f"client_{CLIENT_ID}.csv"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing client manifest: {manifest_path}")

    for round_id in range(1, rounds + 1):
        assignment = get_assignment(round_id)

        if assignment["status"] == "done":
            print(f"[client {CLIENT_ID}] server says run is done", flush=True)
            return

        if not assignment["selected"]:
            print(f"[client {CLIENT_ID}] not selected for round {round_id}", flush=True)
            # Wait for the round to finish before checking next round.
            while True:
                status = requests.get(
                    f"{FED_SERVER_URL}/round/{round_id}/status",
                    timeout=10,
                ).json()
                if status.get("global_checkpoint_exists"):
                    break
                time.sleep(3)
            continue

        print(f"[client {CLIENT_ID}] training round {round_id}", flush=True)

        update_path = UPDATES_DIR / f"client_{CLIENT_ID}_round_{round_id}.pt"

        update_path, num_samples, local_loss, local_acc = train_one_client_round(
            client_id=CLIENT_ID,
            round_id=round_id,
            model_name=assignment["model_name"],
            num_classes=int(assignment["num_classes"]),
            global_state_path=assignment["global_state_path"],
            client_manifest_path=str(manifest_path),
            output_update_path=str(update_path),
            batch_size=batch_size,
            local_epochs=int(assignment["local_epochs"]),
            lr=float(assignment["lr"]),
            pretrained=False,
            preserve_local_classifier=(
                weight_method == "paper_celm" and round_id <= paper_warmup_rounds
            ),
            local_classifier_state_path=str(local_classifier_path),
        )

        payload = {
            "client_id": CLIENT_ID,
            "round_id": round_id,
            "num_samples": num_samples,
            "update_path": update_path,
            "local_loss": local_loss,
            "local_accuracy": local_acc,
        }

        r = requests.post(
            f"{FED_SERVER_URL}/submit_update",
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        print(
            f"[client {CLIENT_ID}] submitted round {round_id}: {r.json()}",
            flush=True,
        )

    print(f"[client {CLIENT_ID}] finished all rounds", flush=True)


if __name__ == "__main__":
    main()
