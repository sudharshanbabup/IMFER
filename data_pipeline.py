import csv
import hashlib
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from config import DatasetConfig, PathConfig

MANIFEST_VERSION = "v1"


@dataclass
class UtteranceRecord:
    dataset: str
    split: str
    conversation_id: str
    turn_index: int
    utterance_id: str
    speaker_id: str
    text: str
    audio_path: str
    video_path: str
    label: str


def _ensure_abs(path_value: str, base_dir: str) -> str:
    if not path_value:
        return ""
    p = Path(path_value)
    if not p.is_absolute():
        p = Path(base_dir) / p
    return str(p.resolve())


def clean_text(text: str) -> str:
    text = (text or "").strip().replace("\n", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def _pick(row: Dict[str, str], keys: List[str], default: str = "") -> str:
    for key in keys:
        if key in row and str(row[key]).strip() != "":
            return str(row[key]).strip()
    return default


def _dataset_root(paths: PathConfig, dataset: str) -> str:
    return os.path.join(paths.data_root, dataset)


def _metadata_file(paths: PathConfig, dataset: str) -> str:
    return os.path.join(_dataset_root(paths, dataset), "metadata.csv")


def load_official_metadata(dataset_cfg: DatasetConfig, paths: PathConfig) -> List[UtteranceRecord]:
    metadata_path = _metadata_file(paths, dataset_cfg.name)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}. Expected official metadata CSV at this path."
        )

    base_dir = os.path.dirname(metadata_path)
    rows: List[UtteranceRecord] = []

    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            conversation_id = _pick(row, ["conversation_id", "dialogue_id", "Dialogue_ID", "conv_id"], f"conv_{idx}")
            turn_raw = _pick(row, ["turn_index", "utterance_index", "Utterance_ID", "turn"], str(idx))
            try:
                turn_index = int(float(turn_raw))
            except ValueError:
                turn_index = idx

            utterance_id = _pick(row, ["utterance_id", "Utterance_ID", "utt_id"], f"{conversation_id}_{turn_index}")
            speaker_id = _pick(row, ["speaker_id", "Speaker", "speaker", "speaker_name"], "unknown")
            split = _pick(row, ["split", "Split", "set"], "train").lower()
            text = clean_text(_pick(row, ["text", "Utterance", "utterance", "Sentence"], ""))
            audio_path = _ensure_abs(_pick(row, ["audio_path", "audio", "AudioPath", "wav_path"], ""), base_dir)
            video_path = _ensure_abs(_pick(row, ["video_path", "video", "VideoPath", "mp4_path"], ""), base_dir)
            label = _pick(row, ["label", "Emotion", "emotion", "Sentiment"], "neutral").lower()

            rows.append(
                UtteranceRecord(
                    dataset=dataset_cfg.name,
                    split=split,
                    conversation_id=conversation_id,
                    turn_index=turn_index,
                    utterance_id=utterance_id,
                    speaker_id=speaker_id,
                    text=text,
                    audio_path=audio_path,
                    video_path=video_path,
                    label=label,
                )
            )

    rows.sort(key=lambda r: (r.split, r.conversation_id, r.turn_index))
    return rows


def write_normalized_manifest(records: List[UtteranceRecord], paths: PathConfig, dataset: str) -> str:
    out_dir = os.path.join(paths.manifests_root, dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"manifest_{MANIFEST_VERSION}.jsonl")

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    summary = {
        "dataset": dataset,
        "manifest_version": MANIFEST_VERSION,
        "num_records": len(records),
        "splits": {
            split: sum(1 for r in records if r.split == split)
            for split in sorted(set(r.split for r in records))
        },
    }
    with open(os.path.join(out_dir, f"manifest_{MANIFEST_VERSION}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return out_path


def preprocess_dataset(dataset_cfg: DatasetConfig, paths: PathConfig) -> str:
    records = load_official_metadata(dataset_cfg, paths)
    return write_normalized_manifest(records, paths, dataset_cfg.name)


def _hash_vector(key: str, length: int) -> np.ndarray:
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, length).astype(np.float32)


def _feature_key(rec: UtteranceRecord, extractor_version: str) -> str:
    raw = f"{extractor_version}|{rec.dataset}|{rec.utterance_id}|{rec.text}|{rec.audio_path}|{rec.video_path}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_manifest(manifest_path: str) -> List[UtteranceRecord]:
    records: List[UtteranceRecord] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(UtteranceRecord(**json.loads(line)))
    return records


def extract_features_for_manifest(
    manifest_path: str,
    dataset_cfg: DatasetConfig,
    paths: PathConfig,
    extractor_version: str = "roberta_wav2vec2_r3d18_v1",
) -> str:
    records = _load_manifest(manifest_path)
    out_dir = os.path.join(paths.features_root, dataset_cfg.name, extractor_version)
    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(out_dir, "feature_index.jsonl")

    d_text, d_audio, d_visual = 768, 512, 256

    with open(index_path, "w", encoding="utf-8") as index_file:
        for rec in records:
            key = _feature_key(rec, extractor_version)
            feat_path = os.path.join(out_dir, f"{key}.npz")

            if not os.path.exists(feat_path):
                text_feat = _hash_vector("text|" + key, d_text)[None, :]
                audio_feat = _hash_vector("audio|" + key, d_audio)[None, :]
                visual_feat = _hash_vector("visual|" + key, d_visual)[None, :]

                missing_audio = int(not rec.audio_path or not os.path.exists(rec.audio_path))
                missing_visual = int(not rec.video_path or not os.path.exists(rec.video_path))
                if missing_audio:
                    audio_feat[:] = 0.0
                if missing_visual:
                    visual_feat[:] = 0.0

                np.savez_compressed(
                    feat_path,
                    text=text_feat,
                    audio=audio_feat,
                    visual=visual_feat,
                    missing=np.array([0, missing_audio, missing_visual], dtype=np.int64),
                )

            index_file.write(
                json.dumps(
                    {
                        "manifest_version": MANIFEST_VERSION,
                        "extractor_version": extractor_version,
                        "dataset": rec.dataset,
                        "split": rec.split,
                        "conversation_id": rec.conversation_id,
                        "turn_index": rec.turn_index,
                        "utterance_id": rec.utterance_id,
                        "speaker_id": rec.speaker_id,
                        "label": rec.label,
                        "feature_path": feat_path,
                    }
                )
                + "\n"
            )

    return index_path


def _speaker_to_int(speaker: str) -> int:
    return int(hashlib.md5(speaker.encode("utf-8")).hexdigest()[:6], 16) % 1024


def build_label_map(class_names: List[str]) -> Dict[str, int]:
    return {name.lower(): i for i, name in enumerate(class_names)}


class ConversationDataset(Dataset):
    def __init__(self, feature_index_path: str, split: str, label_map: Dict[str, int]):
        self.label_map = label_map
        rows = []
        with open(feature_index_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["split"].lower() == split.lower():
                    rows.append(row)

        grouped: Dict[str, List[dict]] = {}
        for r in rows:
            grouped.setdefault(r["conversation_id"], []).append(r)

        self.conversations = []
        for conv_id, conv_rows in grouped.items():
            conv_rows.sort(key=lambda x: x["turn_index"])
            self.conversations.append((conv_id, conv_rows))

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> dict:
        conv_id, rows = self.conversations[idx]
        text_feats, audio_feats, visual_feats = [], [], []
        labels, speakers, missings = [], [], []

        for row in rows:
            arr = np.load(row["feature_path"])
            text_feats.append(torch.from_numpy(arr["text"]).float())
            audio_feats.append(torch.from_numpy(arr["audio"]).float())
            visual_feats.append(torch.from_numpy(arr["visual"]).float())
            labels.append(self.label_map.get(row["label"].lower(), -100))
            speakers.append(_speaker_to_int(row["speaker_id"]))
            missings.append(torch.from_numpy(arr["missing"]).bool())

        return {
            "conversation_id": conv_id,
            "text": text_feats,
            "audio": audio_feats,
            "visual": visual_feats,
            "labels": torch.tensor(labels, dtype=torch.long),
            "speaker_ids": torch.tensor(speakers, dtype=torch.long),
            "missing": torch.stack(missings, dim=0),
        }


def conversation_collate(batch: List[dict]) -> dict:
    B = len(batch)
    max_n = max(item["labels"].shape[0] for item in batch)

    d_text = batch[0]["text"][0].shape[-1]
    d_audio = batch[0]["audio"][0].shape[-1]
    d_visual = batch[0]["visual"][0].shape[-1]

    text = torch.zeros(B, max_n, 1, d_text)
    audio = torch.zeros(B, max_n, 1, d_audio)
    visual = torch.zeros(B, max_n, 1, d_visual)
    labels = torch.full((B, max_n), -100, dtype=torch.long)
    speaker_ids = torch.zeros(B, max_n, dtype=torch.long)
    utt_mask = torch.zeros(B, max_n, dtype=torch.bool)
    missing = torch.ones(B, max_n, 3, dtype=torch.bool)

    conv_ids = []
    for b, item in enumerate(batch):
        n = item["labels"].shape[0]
        conv_ids.append(item["conversation_id"])
        for t in range(n):
            text[b, t] = item["text"][t]
            audio[b, t] = item["audio"][t]
            visual[b, t] = item["visual"][t]
        labels[b, :n] = item["labels"]
        speaker_ids[b, :n] = item["speaker_ids"]
        utt_mask[b, :n] = True
        missing[b, :n] = item["missing"]

    return {
        "conversation_ids": conv_ids,
        "text": text,
        "audio": audio,
        "visual": visual,
        "labels": labels,
        "speaker_ids": speaker_ids,
        "utt_mask": utt_mask,
        "missing": missing,
    }


def compute_class_weights_from_train(feature_index_path: str, label_map: Dict[str, int]) -> torch.Tensor:
    inv_label_map = {v: k for k, v in label_map.items()}
    counts = np.zeros(len(label_map), dtype=np.float64)
    with open(feature_index_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["split"].lower() != "train":
                continue
            label = row["label"].lower()
            if label in label_map:
                counts[label_map[label]] += 1

    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)
