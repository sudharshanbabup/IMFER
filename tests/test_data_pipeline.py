import csv
import os
import tempfile
import unittest
import importlib.util

from config import IMFERConfig, IEMOCAP


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for data pipeline tests")
class DataPipelineTests(unittest.TestCase):
    def _write_metadata(self, root: str):
        ds = os.path.join(root, "iemocap")
        os.makedirs(ds, exist_ok=True)
        path = os.path.join(ds, "metadata.csv")
        rows = [
            ["train", "c1", "0", "u1", "spk1", "hello", "", "", "happy"],
            ["train", "c1", "1", "u2", "spk2", "world", "", "", "sad"],
            ["val", "c2", "0", "u3", "spk1", "foo", "", "", "neutral"],
            ["test", "c3", "0", "u4", "spk3", "bar", "", "", "angry"],
        ]
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["split", "conversation_id", "turn_index", "utterance_id", "speaker_id", "text", "audio_path", "video_path", "label"])
            writer.writerows(rows)

    def test_schema_split_label_and_smoke(self):
        import torch
        from data_pipeline import (
            ConversationDataset,
            build_label_map,
            conversation_collate,
            extract_features_for_manifest,
            load_official_metadata,
            preprocess_dataset,
        )

        with tempfile.TemporaryDirectory() as tmp:
            self._write_metadata(tmp)
            cfg = IMFERConfig(dataset=IEMOCAP)
            cfg.paths.data_root = tmp
            cfg.paths.manifests_root = os.path.join(tmp, "manifests")
            cfg.paths.features_root = os.path.join(tmp, "features")

            records = load_official_metadata(cfg.dataset, cfg.paths)
            self.assertEqual(4, len(records))
            self.assertEqual({"train", "val", "test"}, set(r.split for r in records))

            label_map = build_label_map(cfg.dataset.class_names)
            self.assertIn("happy", label_map)
            self.assertIn("angry", label_map)

            manifest = preprocess_dataset(cfg.dataset, cfg.paths)
            feature_index = extract_features_for_manifest(manifest, cfg.dataset, cfg.paths)

            train_ds = ConversationDataset(feature_index, "train", label_map)
            self.assertGreaterEqual(len(train_ds), 1)

            batch = conversation_collate([train_ds[0]])
            self.assertEqual(batch["text"].shape[0], 1)
            self.assertEqual(batch["labels"].shape[0], 1)
            self.assertEqual(batch["utt_mask"].dtype, torch.bool)


if __name__ == "__main__":
    unittest.main()
