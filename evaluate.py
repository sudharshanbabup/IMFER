import argparse
import csv
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


def weighted_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    from collections import Counter

    counts = Counter(y_true)
    N = len(y_true)
    if N == 0:
        return 0.0

    wf1 = 0.0
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        wf1 += (counts.get(c, 0) / N) * f1
    return wf1 * 100


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    if len(y_true) == 0:
        return 0.0
    f1_scores = []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_scores.append(2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
    return float(np.mean(f1_scores) * 100)


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict[str, float]:
    out = {}
    for c, name in enumerate(class_names):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        out[name] = float((2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0) * 100)
    return out


def paired_t_test(scores_ours: List[float], scores_baseline: List[float]) -> Dict[str, float]:
    ours = np.array(scores_ours)
    base = np.array(scores_baseline)
    t_stat, p_value = stats.ttest_rel(ours, base)
    diff = ours - base
    d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0
    mean = np.mean(ours)
    se = np.std(ours, ddof=1) / np.sqrt(len(ours)) if len(ours) > 1 else 0.0
    return {
        "mean": float(mean),
        "std": float(np.std(ours, ddof=1)) if len(ours) > 1 else 0.0,
        "ci_95": float(1.96 * se),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(d),
    }


def _read_predictions_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true.append(int(row["y_true"]))
            y_pred.append(int(row["y_pred"]))
    return np.asarray(y_true), np.asarray(y_pred)


def evaluate_prediction_file(path: str, num_classes: int) -> Dict[str, float]:
    y_true, y_pred = _read_predictions_csv(path)
    return {
        "wf1": weighted_f1(y_true, y_pred, num_classes),
        "mf1": macro_f1(y_true, y_pred, num_classes),
        "accuracy": float(100.0 * np.mean(y_true == y_pred)) if y_true.size else 0.0,
        "num_samples": int(y_true.size),
    }


def evaluate_artifacts(artifacts_root: str, dataset: str, num_classes: int) -> Dict:
    pred_files = sorted(glob.glob(os.path.join(artifacts_root, dataset, "seed_*", "predictions", "*.csv")))
    run_metrics = []
    for path in pred_files:
        seed = int(path.split("seed_")[1].split(os.sep)[0])
        m = evaluate_prediction_file(path, num_classes)
        m["seed"] = seed
        m["prediction_file"] = path
        run_metrics.append(m)

    wf1 = np.array([m["wf1"] for m in run_metrics], dtype=np.float64)
    summary = {
        "dataset": dataset,
        "num_runs": len(run_metrics),
        "wf1_mean": float(wf1.mean()) if wf1.size else 0.0,
        "wf1_std": float(wf1.std(ddof=1)) if wf1.size > 1 else 0.0,
        "wf1_ci95": float(1.96 * wf1.std(ddof=1) / np.sqrt(wf1.size)) if wf1.size > 1 else 0.0,
        "runs": run_metrics,
    }
    return summary


def compute_aopc_from_steps(steps_csv: str) -> Dict[str, float]:
    wf1_steps = []
    with open(steps_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wf1_steps.append(float(row["wf1"]))

    if not wf1_steps:
        return {"aopc": 0.0, "drops": []}

    baseline = wf1_steps[0]
    drops = [baseline - v for v in wf1_steps]
    return {"aopc": float(np.mean(drops)), "drops": drops}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Artifact-driven IMFER evaluation")
    parser.add_argument("--artifacts_root", type=str, default="./artifacts")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--aopc_steps_csv", type=str, default="")
    args = parser.parse_args()

    summary = evaluate_artifacts(args.artifacts_root, args.dataset, args.num_classes)
    out_path = os.path.join(args.artifacts_root, args.dataset, "aggregate", "evaluation_summary.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

    if args.aopc_steps_csv:
        aopc = compute_aopc_from_steps(args.aopc_steps_csv)
        aopc_path = os.path.join(args.artifacts_root, args.dataset, "aggregate", "aopc.json")
        with open(aopc_path, "w", encoding="utf-8") as f:
            json.dump(aopc, f, indent=2)
        print(json.dumps(aopc, indent=2))
