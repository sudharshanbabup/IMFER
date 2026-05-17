import csv
import json
import os
from datetime import datetime
from typing import Dict, Iterable, List


def _ts() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def prepare_run_dirs(root: str, dataset: str, seed: int) -> Dict[str, str]:
    run_root = os.path.join(root, dataset, f"seed_{seed}")
    dirs = {
        "run_root": run_root,
        "checkpoints": os.path.join(run_root, "checkpoints"),
        "predictions": os.path.join(run_root, "predictions"),
        "logs": os.path.join(run_root, "logs"),
        "metrics": os.path.join(run_root, "metrics"),
    }
    for p in dirs.values():
        os.makedirs(p, exist_ok=True)
    return dirs


def save_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_metrics_json(metrics_dir: str, name: str, metrics: Dict):
    save_json(os.path.join(metrics_dir, f"{name}.json"), metrics)


def save_predictions_csv(predictions_dir: str, rows: List[Dict], name: str = "predictions") -> str:
    path = os.path.join(predictions_dir, f"{name}.csv")
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("conversation_id,turn_index,y_true,y_pred,mcs_text,mcs_audio,mcs_visual\n")
        return path

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def append_run_log(logs_dir: str, message: str, name: str = "train.log"):
    path = os.path.join(logs_dir, name)
    with open(path, "a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def save_aggregate_metrics(root: str, dataset: str, run_rows: List[Dict]) -> str:
    out_dir = os.path.join(root, dataset, "aggregate")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "metrics.csv")
    if run_rows:
        fields = list(run_rows[0].keys())
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(run_rows)
    return path
