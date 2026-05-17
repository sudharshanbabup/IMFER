import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_curve(csv_path: str) -> List[float]:
    vals = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vals.append(float(row["wf1"]))
    return vals


def auc_from_curve(curve: List[float], base_wf1: float) -> float:
    if not curve or base_wf1 == 0:
        return 0.0
    curve_norm = [v / base_wf1 for v in curve]
    return float(np.trapz(curve_norm, dx=1.0 / max(1, len(curve_norm) - 1)))


def summarize_curves(deletion_csv: str, insertion_csv: str) -> Dict[str, float]:
    deletion = _load_curve(deletion_csv)
    insertion = _load_curve(insertion_csv)
    base = deletion[0] if deletion else (insertion[-1] if insertion else 1.0)
    return {
        "deletion_auc": auc_from_curve(deletion, base),
        "insertion_auc": auc_from_curve(insertion, base),
        "deletion_steps": deletion,
        "insertion_steps": insertion,
    }


def plot_curves(deletion_csv: str, insertion_csv: str, save_path: str):
    deletion = _load_curve(deletion_csv)
    insertion = _load_curve(insertion_csv)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(range(len(deletion)), deletion, marker="o")
    ax[0].set_title("Deletion Curve")
    ax[0].set_xlabel("k")
    ax[0].set_ylabel("WF1")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(range(len(insertion)), insertion, marker="o")
    ax[1].set_title("Insertion Curve")
    ax[1].set_xlabel("k")
    ax[1].set_ylabel("WF1")
    ax[1].grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Artifact-driven insertion/deletion evaluation")
    parser.add_argument("--deletion_csv", type=str, required=True)
    parser.add_argument("--insertion_csv", type=str, required=True)
    parser.add_argument("--out_json", type=str, default="")
    parser.add_argument("--fig_path", type=str, default="")
    args = parser.parse_args()

    out = summarize_curves(args.deletion_csv, args.insertion_csv)
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    if args.fig_path:
        plot_curves(args.deletion_csv, args.insertion_csv, args.fig_path)

    print(json.dumps(out, indent=2))
