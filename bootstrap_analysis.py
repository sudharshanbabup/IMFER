import argparse
import csv
import json
import os
from typing import Tuple

import numpy as np


def bootstrap_ci(scores: np.ndarray, n_bootstrap: int = 10000, ci_level: float = 0.95, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    n = len(scores)
    if n == 0:
        return 0.0, 0.0, 0.0
    boot_means = np.array([rng.choice(scores, size=n, replace=True).mean() for _ in range(n_bootstrap)])
    alpha = 1 - ci_level
    return float(scores.mean()), float(np.percentile(boot_means, 100 * alpha / 2)), float(np.percentile(boot_means, 100 * (1 - alpha / 2)))


def load_wf1(aggregate_csv: str) -> np.ndarray:
    vals = []
    with open(aggregate_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vals.append(float(row["wf1"]))
    return np.array(vals, dtype=np.float64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap analysis from artifact aggregate metrics")
    parser.add_argument("--aggregate_csv", type=str, required=True)
    parser.add_argument("--out_json", type=str, default="")
    args = parser.parse_args()

    wf1 = load_wf1(args.aggregate_csv)
    mean, lo, hi = bootstrap_ci(wf1)
    out = {
        "num_runs": int(len(wf1)),
        "wf1_mean": mean,
        "wf1_std": float(wf1.std(ddof=1)) if len(wf1) > 1 else 0.0,
        "bootstrap_ci95": [lo, hi],
    }

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
