"""Experiment specifications, grid construction, and directory naming."""

from dataclasses import dataclass
from itertools import product
from math import ceil
from pathlib import Path
import numpy as np #type: ignore


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: int
    clients: int
    distribution: str
    participation: float
    method: str

    @property
    def clients_per_round(self):
        return max(1, int(ceil(self.participation * self.clients)))


def build_experiment_grid(client_counts, distributions, participation_rates, methods):
    return [
        ExperimentSpec(index, clients, distribution, float(participation), method)
        for index, (clients, distribution, participation, method)
        in enumerate(product(client_counts, distributions, participation_rates, methods), start=1)
    ]


def slug(value):
    return "".join(character.lower() if character.isalnum() else "_" for character in str(value)).strip("_").replace("__", "_")


def experiment_directory(base_dir, spec, source_name):
    tag = (f"exp{spec.experiment_id:02d}_{slug(source_name)}_clients{spec.clients}_"
           f"{slug(spec.distribution.replace('(', '').replace(')', ''))}_"
           f"participation{int(spec.participation * 100)}_{slug(spec.method)}")
    path = Path(base_dir) / tag
    path.mkdir(parents=True, exist_ok=True)
    return path


def subset_clients_for_participation(client_loaders, participation_rate, seed=42):
    if participation_rate >= 0.999:
        return client_loaders
    rng = np.random.RandomState(seed)
    count = max(1, int(ceil(participation_rate * len(client_loaders))))
    indices = sorted(rng.choice(len(client_loaders), size=count, replace=False).tolist())
    return [client_loaders[index] for index in indices]


_slug = slug
_exp_dir = lambda exp_id, source_name, n_clients, distribution, participation, method: str(
    experiment_directory(Path("results") / "experiments", ExperimentSpec(exp_id, n_clients, distribution, participation, method), source_name)
)
