# IMFER: Interpretable Multimodal Fusion for Emotion Recognition

This repository now supports an artifact-driven, real-data reproducibility workflow for:
- IEMOCAP
- MELD
- EmoryNLP

## Reproducibility target matrix

| Dataset | Splits | Labels | Primary metrics | Paper outputs targeted |
|---|---|---|---|---|
| IEMOCAP | train/val/test | happy, sad, neutral, angry, excited, frustrated | WF1, MF1, Acc, per-class F1 | Table II/IV/VI/VII, Fig. 3/5/6/7/11/12 |
| MELD | train/dev/test | neutral, surprise, fear, sadness, joy, disgust, anger | WF1, MF1, Acc | Table II, Fig. 3/9 |
| EmoryNLP | train/dev/test | joyful, peaceful, powerful, scared, mad, sad, neutral | WF1, MF1 | Table II |

## Dataset sources

Use the original or official dataset sources below before preparing `metadata.csv` files:

- **IEMOCAP**: USC SAIL IEMOCAP release portal: `https://sail.usc.edu/iemocap/`  
  Release/download form: `https://sail.usc.edu/iemocap/iemocap_release.htm`
- **MELD**: official dataset repository: `https://github.com/declare-lab/MELD`  
  Raw download noted by the dataset authors: `http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz`
- **EmoryNLP Emotion Detection**: official repository: `https://github.com/emorynlp/emotion-detection`

## Expected metadata input

After downloading the datasets, convert or organize their annotations into the following normalized metadata files:
- `./data/iemocap/metadata.csv`
- `./data/meld/metadata.csv`
- `./data/emorynlp/metadata.csv`

Each CSV should include fields mappable to:
- `split`
- `conversation_id`
- `turn_index`
- `utterance_id`
- `speaker_id`
- `text`
- `audio_path`
- `video_path`
- `label`

## Dataset preparation notes

- **IEMOCAP**: the release contains audio, video, transcripts, and annotations, but this repo expects you to normalize those files into a single `metadata.csv` under `./data/iemocap/`.
- **MELD**: use the official split naming (`train`, `dev`, `test`) when generating metadata. This repo will automatically use `dev` as the validation split for MELD.
- **EmoryNLP**: use the official split naming (`train`, `dev`, `test`) when generating metadata. If audio/video assets are unavailable in your environment, the pipeline can still run with empty paths, but missing modalities will be zero-filled during feature generation.

## Pipeline overview

1. **Preprocess + normalize manifest** (`data_pipeline.preprocess_dataset`)  
   Produces: `./data/manifests/<dataset>/manifest_v1.jsonl`
2. **Feature extraction + deterministic caching** (`extract_features_for_manifest`)  
   Produces: `./data/features/<dataset>/roberta_wav2vec2_r3d18_v1/`
3. **Conversation dataloading** (`ConversationDataset` + `conversation_collate`)  
   Provides padded conversation tensors, utterance masks, and missing-modality flags
4. **Training** (`train.py`)  
   Uses real dataloaders, true conversation CASGT batching, per-dataset labels, MELD class weighting from train priors
5. **Artifact-driven evaluation** (`evaluate.py`, `robustness.py`, `insertion_deletion.py`, `bootstrap_analysis.py`, `visualize_results.py`)

## Commands

### Train
```bash
python train.py --dataset iemocap --device cpu
python train.py --dataset meld --device cpu
python train.py --dataset emorynlp --device cpu
```

### Aggregate evaluation from saved predictions
```bash
python evaluate.py \
  --artifacts_root ./artifacts \
  --dataset iemocap \
  --num_classes 6
```

### Bootstrap analysis
```bash
python bootstrap_analysis.py \
  --aggregate_csv ./artifacts/iemocap/aggregate/metrics.csv
```

### Generate figures from aggregate metrics
```bash
python visualize_results.py \
  --aggregate_csv ./artifacts/iemocap/aggregate/metrics.csv \
  --output_dir ./figures
```

## Artifact layout

```
artifacts/
  <dataset>/
    seed_<seed>/
      checkpoints/best.pt
      predictions/test_predictions.csv
      logs/train.log
      metrics/test_metrics.json
    aggregate/
      metrics.csv
      summary.json
      evaluation_summary.json
      aopc.json   # optional, if provided
```

## Validation checkpoints

- Schema + split + label-map checks: `tests/test_data_pipeline.py`
- Smoke run: one-seed training with small metadata subset
- Full reproduction: single-seed then 5-seed

## MCS formulation note

`MCSLayer` computes normalized modality energies from per-modality projections for each utterance and is aligned to the modality-energy normalization definition used in the paper. Any residual approximation error comes from interaction terms introduced by gating and contextualization.

## Known blockers and fallback decisions

- Some environments may not have full raw audio/video assets available.
- Official split files can differ by release (`val` vs `dev` naming).
- Certain preprocessing details in the paper may be under-specified.
- Exact metric matching can be limited without author-released checkpoints.

## License

For academic research purposes.
