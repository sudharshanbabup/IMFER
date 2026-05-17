from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ReproTarget:
    dataset: str
    splits: List[str]
    labels: List[str]
    metrics: List[str]
    paper_outputs: List[str]


REPRODUCIBILITY_MATRIX: Dict[str, ReproTarget] = {
    "iemocap": ReproTarget(
        dataset="iemocap",
        splits=["train", "val", "test"],
        labels=["happy", "sad", "neutral", "angry", "excited", "frustrated"],
        metrics=["wf1", "mf1", "accuracy", "per_class_f1"],
        paper_outputs=["Table II", "Table IV", "Table VI", "Table VII", "Fig. 3", "Fig. 5", "Fig. 6", "Fig. 7", "Fig. 11", "Fig. 12"],
    ),
    "meld": ReproTarget(
        dataset="meld",
        splits=["train", "dev", "test"],
        labels=["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"],
        metrics=["wf1", "mf1", "accuracy"],
        paper_outputs=["Table II", "Fig. 3", "Fig. 9"],
    ),
    "emorynlp": ReproTarget(
        dataset="emorynlp",
        splits=["train", "dev", "test"],
        labels=["joyful", "peaceful", "powerful", "scared", "mad", "sad", "neutral"],
        metrics=["wf1", "mf1"],
        paper_outputs=["Table II"],
    ),
}


BLOCKERS = [
    "Raw audio/video access may be unavailable for some datasets in local environment.",
    "Official split definitions can differ across released variants (val/dev naming mismatches).",
    "Paper preprocessing details may be ambiguous for tokenizer normalization and clip boundaries.",
    "Absence of released checkpoints may prevent exact-number reproduction.",
]
