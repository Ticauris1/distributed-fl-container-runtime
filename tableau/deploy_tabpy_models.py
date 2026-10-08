"""Deploy the project's named calculation functions to a running TabPy server."""

import argparse

from tabpy.tabpy_tools.client import Client

from tableau.tabpy_functions import (
    confidence_interval_95,
    mean_value,
    paired_delta,
    pareto_score,
    rolling_mean,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:9004/")
    parser.add_argument("--username")
    parser.add_argument("--password")
    args = parser.parse_args()
    client = Client(args.endpoint)
    if args.username:
        client.set_credentials(args.username, args.password or "")
    functions = {
        "fl_mean": mean_value,
        "fl_confidence_interval_95": confidence_interval_95,
        "fl_paired_delta": paired_delta,
        "fl_rolling_mean": rolling_mean,
        "fl_pareto_score": pareto_score,
    }
    for name, function in functions.items():
        client.deploy(name, function, override=True)
        print(f"[tabpy] deployed {name}")


if __name__ == "__main__":
    main()
