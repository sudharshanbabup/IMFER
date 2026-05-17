import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from artifacts import (
    append_run_log,
    prepare_run_dirs,
    save_aggregate_metrics,
    save_metrics_json,
    save_predictions_csv,
)
from config import EMORYNLP, IEMOCAP, MELD, IMFERConfig
from data_pipeline import (
    ConversationDataset,
    build_label_map,
    compute_class_weights_from_train,
    conversation_collate,
    extract_features_for_manifest,
    preprocess_dataset,
)
from evaluate import macro_f1, weighted_f1
from losses import IMFERLoss
from models import IMFER


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer(model: IMFER, cfg: IMFERConfig):
    return optim.AdamW(model.parameters(), lr=cfg.train.lr_new, weight_decay=cfg.train.weight_decay)


def get_scheduler(optimizer, num_training_steps: int, warmup_fraction: float):
    num_warmup_steps = int(num_training_steps * warmup_fraction)

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _build_loaders(cfg: IMFERConfig, batch_size: int):
    manifest_path = preprocess_dataset(cfg.dataset, cfg.paths)
    feature_index_path = extract_features_for_manifest(manifest_path, cfg.dataset, cfg.paths)
    label_map = build_label_map(cfg.dataset.class_names)

    train_ds = ConversationDataset(feature_index_path, "train", label_map)
    val_split = "dev" if cfg.dataset.name in {"meld", "emorynlp"} else "val"
    val_ds = ConversationDataset(feature_index_path, val_split, label_map)
    test_ds = ConversationDataset(feature_index_path, "test", label_map)

    if len(val_ds) == 0:
        val_ds = train_ds

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=conversation_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=conversation_collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=conversation_collate)

    class_weights = None
    if cfg.dataset.name == "meld":
        class_weights = compute_class_weights_from_train(feature_index_path, label_map)

    return train_loader, val_loader, test_loader, label_map, class_weights


def _to_device(batch: dict, device: torch.device) -> dict:
    out = dict(batch)
    for key in ["text", "audio", "visual", "labels", "speaker_ids", "utt_mask", "missing"]:
        out[key] = out[key].to(device)
    return out


def train_one_epoch(model, criterion, optimizer, scheduler, train_loader, device):
    model.train()
    total = {"loss": 0.0, "ce": 0.0, "mcs": 0.0, "align": 0.0}
    n_batches = 0

    for batch in train_loader:
        batch = _to_device(batch, device)
        optimizer.zero_grad()

        out = model(batch["text"], batch["audio"], batch["visual"], batch["speaker_ids"], batch["utt_mask"])
        valid = batch["utt_mask"] & (batch["labels"] >= 0)

        logits = out["logits"][valid]
        labels = batch["labels"][valid]
        mcs_scores = out["mcs_scores"][valid]
        z_text = out["modality_utts"]["text"][valid]
        z_audio = out["modality_utts"]["audio"][valid]

        losses = criterion(logits, labels, mcs_scores, z_text, z_audio)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total["loss"] += losses["total"].item()
        total["ce"] += losses["ce"].item()
        total["mcs"] += losses["mcs"].item()
        total["align"] += losses["align"].item()
        n_batches += 1

    return {k: v / max(1, n_batches) for k, v in total.items()}


@torch.no_grad()
def evaluate_model(model, loader, num_classes, device):
    model.eval()
    y_true, y_pred = [], []
    mcs_all = []
    pred_rows = []

    for batch in loader:
        batch_cpu = batch
        batch = _to_device(batch, device)
        out = model(batch["text"], batch["audio"], batch["visual"], batch["speaker_ids"], batch["utt_mask"])

        pred = out["logits"].argmax(dim=-1).cpu()
        labels = batch_cpu["labels"]
        valid = batch_cpu["utt_mask"] & (labels >= 0)

        mcs = out["mcs_scores"].cpu()
        for b in range(labels.shape[0]):
            conv = batch_cpu["conversation_ids"][b]
            for t in range(labels.shape[1]):
                if not bool(valid[b, t]):
                    continue
                yt = int(labels[b, t].item())
                yp = int(pred[b, t].item())
                y_true.append(yt)
                y_pred.append(yp)
                m = mcs[b, t].tolist()
                mcs_all.append(m)
                pred_rows.append(
                    {
                        "conversation_id": conv,
                        "turn_index": t,
                        "y_true": yt,
                        "y_pred": yp,
                        "mcs_text": m[0],
                        "mcs_audio": m[1],
                        "mcs_visual": m[2],
                    }
                )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.size == 0:
        return {
            "wf1": 0.0,
            "mf1": 0.0,
            "accuracy": 0.0,
            "mcs_text": 0.0,
            "mcs_audio": 0.0,
            "mcs_visual": 0.0,
            "rows": pred_rows,
        }

    mcs_np = np.asarray(mcs_all)
    return {
        "wf1": weighted_f1(y_true, y_pred, num_classes),
        "mf1": macro_f1(y_true, y_pred, num_classes),
        "accuracy": float(100.0 * np.mean(y_true == y_pred)),
        "mcs_text": float(mcs_np[:, 0].mean()),
        "mcs_audio": float(mcs_np[:, 1].mean()),
        "mcs_visual": float(mcs_np[:, 2].mean()),
        "rows": pred_rows,
    }


def train_single_run(seed: int, cfg: IMFERConfig, device: torch.device) -> Dict[str, float]:
    set_seed(seed)
    run_dirs = prepare_run_dirs(cfg.paths.artifacts_root, cfg.dataset.name, seed)

    train_loader, val_loader, test_loader, _, class_weights = _build_loaders(cfg, cfg.train.batch_size)

    model = IMFER(
        d_text=cfg.model.d_text,
        d_audio=cfg.model.d_audio,
        d_visual=cfg.model.d_visual,
        d_k=cfg.model.d_k,
        d_model=cfg.model.d_model,
        num_classes=cfg.dataset.num_classes,
        casgt_heads=cfg.model.casgt_heads,
        casgt_layers=cfg.model.casgt_layers,
        context_window=cfg.model.context_window,
        dropout=cfg.model.dropout,
    ).to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)

    criterion = IMFERLoss(
        num_classes=cfg.dataset.num_classes,
        lambda_1=cfg.train.lambda_1,
        lambda_2=cfg.train.lambda_2,
        tau=cfg.train.tau,
        class_weights=class_weights,
    )

    optimizer = get_optimizer(model, cfg)
    num_training_steps = max(1, len(train_loader) * cfg.train.max_epochs)
    scheduler = get_scheduler(optimizer, num_training_steps, cfg.train.warmup_fraction)

    best_val = -1.0
    best_state = None
    patience = 0

    for epoch in range(cfg.train.max_epochs):
        train_metrics = train_one_epoch(model, criterion, optimizer, scheduler, train_loader, device)
        val_metrics = evaluate_model(model, val_loader, cfg.dataset.num_classes, device)

        line = (
            f"epoch={epoch+1} loss={train_metrics['loss']:.4f} ce={train_metrics['ce']:.4f} "
            f"mcs={train_metrics['mcs']:.4f} align={train_metrics['align']:.4f} val_wf1={val_metrics['wf1']:.2f}"
        )
        append_run_log(run_dirs["logs"], line)

        if val_metrics["wf1"] > best_val:
            best_val = val_metrics["wf1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(run_dirs["checkpoints"], "best.pt"))
            patience = 0
        else:
            patience += 1
            if patience >= cfg.train.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_model(model, test_loader, cfg.dataset.num_classes, device)
    pred_path = save_predictions_csv(run_dirs["predictions"], test_metrics.pop("rows"), name="test_predictions")
    save_metrics_json(run_dirs["metrics"], "test_metrics", test_metrics)

    test_metrics["prediction_file"] = pred_path
    return test_metrics


def run_experiment(cfg: IMFERConfig, device: torch.device):
    rows = []
    for seed in cfg.train.seeds:
        result = train_single_run(seed, cfg, device)
        rows.append(
            {
                "seed": seed,
                "wf1": result["wf1"],
                "mf1": result["mf1"],
                "accuracy": result["accuracy"],
                "mcs_text": result["mcs_text"],
                "mcs_audio": result["mcs_audio"],
                "mcs_visual": result["mcs_visual"],
                "prediction_file": result["prediction_file"],
            }
        )

    save_aggregate_metrics(cfg.paths.artifacts_root, cfg.dataset.name, rows)

    wf1 = np.array([r["wf1"] for r in rows], dtype=np.float64)
    summary = {
        "dataset": cfg.dataset.name,
        "num_runs": len(rows),
        "wf1_mean": float(wf1.mean()) if wf1.size else 0.0,
        "wf1_std": float(wf1.std(ddof=1)) if wf1.size > 1 else 0.0,
        "wf1_ci95": float(1.96 * wf1.std(ddof=1) / np.sqrt(wf1.size)) if wf1.size > 1 else 0.0,
        "runs": rows,
    }

    out_summary = os.path.join(cfg.paths.artifacts_root, cfg.dataset.name, "aggregate", "summary.json")
    os.makedirs(os.path.dirname(out_summary), exist_ok=True)
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMFER real-data training")
    parser.add_argument("--dataset", type=str, default="iemocap", choices=["iemocap", "meld", "emorynlp"])
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    dataset_map = {"iemocap": IEMOCAP, "meld": MELD, "emorynlp": EMORYNLP}
    cfg = IMFERConfig(dataset=dataset_map[args.dataset])
    run_experiment(cfg, torch.device(args.device))
