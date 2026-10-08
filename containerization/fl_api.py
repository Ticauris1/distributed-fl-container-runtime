"""
Small HTTP API used by the distributed FL containers.

The server owns:
- global model checkpoints
- round assignments
- aggregation
- validation metrics

Clients own:
- one local data shard
- local training
- update submission
"""

from __future__ import annotations

from typing import Dict, Optional
from pydantic import BaseModel


class RoundAssignment(BaseModel):
    status: str
    round_id: int
    client_id: int
    selected: bool
    global_state_path: Optional[str] = None
    model_name: Optional[str] = None
    source_name: Optional[str] = None
    num_classes: Optional[int] = None
    local_epochs: Optional[int] = None
    lr: Optional[float] = None


class ClientUpdate(BaseModel):
    client_id: int
    round_id: int
    num_samples: int
    update_path: str
    local_loss: float = 0.0
    local_accuracy: float = 0.0


class ServerState(BaseModel):
    current_round: int = 0
    status: str = "initializing"
    message: str = ""
