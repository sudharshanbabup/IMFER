"""
config.py – All hyperparameters and paths for the IMFER framework.

References:
  - Section IV-C (Implementation Details): d_k=64, d=512, W=10, etc.
  - Eq. (7): lambda_1=0.1, lambda_2=0.05, tau=0.07
  - Section IV-A: Dataset splits and class counts
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DatasetConfig:
    """Dataset-specific configuration."""
    name: str
    num_classes: int
    class_names: List[str]
    num_utterances: int  # approximate total utterances
    text_max_len: int = 128       # max BPE tokens (Section IV-C)
    audio_sample_rate: int = 16000  # 16 kHz resampling (Section IV-C)
    video_fps: int = 30           # 30 fps extraction (Section IV-C)
    video_clip_frames: int = 16   # 16-frame clips (Section IV-C)


# ── Dataset definitions (Section IV-A) ──────────────────────────────
IEMOCAP = DatasetConfig(
    name="iemocap",
    num_classes=6,
    class_names=["happy", "sad", "neutral", "angry", "excited", "frustrated"],
    num_utterances=5531,
)

MELD = DatasetConfig(
    name="meld",
    num_classes=7,
    class_names=[
        "neutral", "surprise", "fear", "sadness",
        "joy", "disgust", "anger"
    ],
    num_utterances=13708,
)

EMORYNLP = DatasetConfig(
    name="emorynlp",
    num_classes=7,
    class_names=[
        "joyful", "peaceful", "powerful", "scared",
        "mad", "sad", "neutral"
    ],
    num_utterances=12606,
)


@dataclass
class ModelConfig:
    """
    Model architecture hyperparameters.
    
    All values correspond to Section III and Section IV-C of the paper:
      - d_k = 64           (Eq. 2-3: low-rank projection dimension)
      - d = 512            (hidden dimension after fusion)
      - M = 3              (number of modalities: text, audio, visual)
      - d_text = 768       (RoBERTa-base output dim, Section III-B)
      - d_audio = 512      (wav2vec 2.0 output dim, Section III-B)
      - d_visual = 256     (3D-ResNet output dim, Section III-B)
      - W = 10             (CASGT context window, Section III-C)
      - num_heads = 8      (CASGT transformer heads)
      - num_layers = 4     (CASGT transformer depth)
      - dropout = 0.3
    """
    # ── Modality encoder output dimensions (Section III-B) ──────────
    d_text: int = 768        # RoBERTa-base hidden size
    d_audio: int = 512       # wav2vec 2.0 feature dim
    d_visual: int = 256      # 3D-ResNet feature dim

    # ── HCMA parameters (Section III-D, Eq. 2-3) ───────────────────
    d_k: int = 64            # low-rank projection dimension
    d_model: int = 512       # fused hidden dimension d
    num_modalities: int = 3  # M = 3 (text, audio, visual)

    # ── CASGT parameters (Section III-C) ────────────────────────────
    context_window: int = 10    # W = 10
    casgt_heads: int = 8        # 8 attention heads
    casgt_layers: int = 4       # 4-layer transformer encoder
    casgt_hidden: int = 512     # same as d_model

    # ── MCS parameters (Section III-E) ──────────────────────────────
    # MCS is computed from classifier weight partitions; no extra params

    # ── Regularization ──────────────────────────────────────────────
    dropout: float = 0.3


@dataclass
class TrainConfig:
    """
    Training hyperparameters (Section IV-C).
    """
    # ── Optimizer: AdamW (Loshchilov & Hutter, 2019) ────────────────
    lr_pretrained: float = 2e-5     # learning rate for pretrained encoders
    lr_new: float = 1e-3            # learning rate for new layers
    warmup_fraction: float = 0.10   # 10% linear warmup
    weight_decay: float = 0.01      # AdamW default

    # ── Loss weights (Eq. 7) ────────────────────────────────────────
    lambda_1: float = 0.1           # MCS entropy regularization weight
    lambda_2: float = 0.05          # contrastive alignment weight
    tau: float = 0.07               # temperature for alignment loss (Eq. 8)

    # ── Training schedule ───────────────────────────────────────────
    batch_size: int = 32            # 32 utterances per batch
    max_epochs: int = 100
    patience: int = 10              # early stopping patience on val WF1
    num_runs: int = 5               # independent runs for reporting

    # ── Seeds used across 5 runs ────────────────────────────────────
    seeds: List[int] = field(
        default_factory=lambda: [42, 123, 256, 512, 1024]
    )


@dataclass
class PathConfig:
    """File paths for data and outputs."""
    data_root: str = "./data"
    output_root: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    figures_dir: str = "./figures"

    # ── Pretrained model identifiers ────────────────────────────────
    roberta_model: str = "roberta-base"
    wav2vec_model: str = "facebook/wav2vec2-base-960h"
    resnet3d_model: str = "r3d_18"   # torchvision 3D-ResNet-18


# ── Convenience: assemble full config ───────────────────────────────
@dataclass
class IMFERConfig:
    """Top-level configuration combining all sub-configs."""
    dataset: DatasetConfig = field(default_factory=lambda: IEMOCAP)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    def __post_init__(self):
        os.makedirs(self.paths.output_root, exist_ok=True)
        os.makedirs(self.paths.checkpoint_dir, exist_ok=True)
        os.makedirs(self.paths.figures_dir, exist_ok=True)


if __name__ == "__main__":
    cfg = IMFERConfig()
    print(f"Dataset      : {cfg.dataset.name}")
    print(f"Num classes  : {cfg.dataset.num_classes}")
    print(f"d_k          : {cfg.model.d_k}")
    print(f"d_model      : {cfg.model.d_model}")
    print(f"lambda_1     : {cfg.train.lambda_1}")
    print(f"lambda_2     : {cfg.train.lambda_2}")
    print(f"Seeds        : {cfg.train.seeds}")
