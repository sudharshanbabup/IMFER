import argparse
import csv
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_missing_modality(csv_path: str) -> Dict[str, float]:
    rows = load_csv(csv_path)
    out = {}
    for row in rows:
        out[row["condition"]] = float(row["wf1"])
    return out


def summarize_noise_sensitivity(csv_path: str) -> Dict[str, List[float]]:
    rows = load_csv(csv_path)
    grouped: Dict[str, List[tuple]] = {}
    for row in rows:
        modality = row["modality"]
        sigma = float(row["sigma"])
        wf1 = float(row["wf1"])
        grouped.setdefault(modality, []).append((sigma, wf1))

    out = {}
    for modality, vals in grouped.items():
        vals.sort(key=lambda x: x[0])
        out[modality] = [v for _, v in vals]
        out[f"{modality}_sigma"] = [s for s, _ in vals]
    return out


def plot_noise_sensitivity(noise_csv: str, save_path: str):
    data = summarize_noise_sensitivity(noise_csv)
    plt.figure(figsize=(7, 5))
    for modality in ["text", "audio", "visual", "all"]:
        sigma_key = f"{modality}_sigma"
        if sigma_key not in data:
            continue
        plt.plot(data[sigma_key], data[modality], marker="o", label=modality)
    plt.xlabel("Noise σ")
    plt.ylabel("WF1 (%)")
    plt.title("Noise Sensitivity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Artifact-driven robustness analysis")
    parser.add_argument("--missing_csv", type=str, default="")
    parser.add_argument("--noise_csv", type=str, default="")
    parser.add_argument("--out_json", type=str, default="")
    parser.add_argument("--noise_fig", type=str, default="")
    args = parser.parse_args()

    payload = {}
    if args.missing_csv:
        payload["missing_modality"] = summarize_missing_modality(args.missing_csv)
    if args.noise_csv:
        payload["noise_sensitivity"] = summarize_noise_sensitivity(args.noise_csv)
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    if args.noise_csv and args.noise_fig:
        plot_noise_sensitivity(args.noise_csv, args.noise_fig)

    print(json.dumps(payload, indent=2))
