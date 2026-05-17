import argparse
import csv
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def _read_run_metrics(aggregate_csv: str):
    rows = []
    with open(aggregate_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def plot_wf1_distribution(aggregate_csv: str, save_path: str):
    rows = _read_run_metrics(aggregate_csv)
    wf1 = [float(r["wf1"]) for r in rows]
    seeds = [str(r["seed"]) for r in rows]

    plt.figure(figsize=(8, 4))
    plt.bar(seeds, wf1, color="#e74c3c", alpha=0.85)
    if wf1:
        mean = np.mean(wf1)
        std = np.std(wf1, ddof=1) if len(wf1) > 1 else 0.0
        plt.axhline(mean, color="black", linestyle="--", label=f"mean={mean:.2f}")
        plt.title(f"WF1 by Seed (mean={mean:.2f}, std={std:.2f})")
    plt.xlabel("Seed")
    plt.ylabel("WF1 (%)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_mcs_average(aggregate_csv: str, save_path: str):
    rows = _read_run_metrics(aggregate_csv)
    if not rows:
        return
    text = np.mean([float(r["mcs_text"]) for r in rows])
    audio = np.mean([float(r["mcs_audio"]) for r in rows])
    visual = np.mean([float(r["mcs_visual"]) for r in rows])

    plt.figure(figsize=(5, 4))
    plt.bar(["text", "audio", "visual"], [text, audio, visual], color=["#d62728", "#1f77b4", "#2ca02c"])
    plt.ylim(0, 1)
    plt.ylabel("Average MCS")
    plt.title("Average Modality Contribution")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Artifact-driven figure generation")
    parser.add_argument("--aggregate_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plot_wf1_distribution(args.aggregate_csv, os.path.join(args.output_dir, "wf1_by_seed.png"))
    plot_mcs_average(args.aggregate_csv, os.path.join(args.output_dir, "mcs_average.png"))
    print(f"Saved figures to {args.output_dir}")
