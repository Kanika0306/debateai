"""
Config for the Fact Verification (NLI) model.
Mirrors the fallacy_classifier config style so the two pipelines stay consistent.
"""
import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # Backbone
    model_name: str = "microsoft/deberta-v3-large"
    max_length: int = 128

    # Labels — unified 3-class verification scheme
    # SUPPORTS   = claim is backed by evidence / rated true
    # REFUTES    = claim is contradicted by evidence / rated false
    # NOT_ENOUGH_INFO = evidence insufficient, or claim only partially true
    labels: List[str] = field(default_factory=lambda: [
        "SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"
    ])

    # LIAR's 6-way truthfulness scale collapsed into the 3-class scheme above.
    # This is an explicit modeling choice — adjust if you want a finer-grained
    # LIAR-only head instead of merging it with FEVER's 3-class NLI task.
    liar_label_map: dict = field(default_factory=lambda: {
        "true": "SUPPORTS",
        "mostly-true": "SUPPORTS",
        "half-true": "NOT_ENOUGH_INFO",
        "barely-true": "NOT_ENOUGH_INFO",
        "false": "REFUTES",
        "pants-fire": "REFUTES",
        0: "REFUTES",
        1: "REFUTES",
        2: "NOT_ENOUGH_INFO",
        3: "NOT_ENOUGH_INFO",
        4: "SUPPORTS",
        5: "SUPPORTS",
        "0": "REFUTES",
        "1": "REFUTES",
        "2": "NOT_ENOUGH_INFO",
        "3": "NOT_ENOUGH_INFO",
        "4": "SUPPORTS",
        "5": "SUPPORTS",
    })

    fever_label_map: dict = field(default_factory=lambda: {
        "SUPPORTS": "SUPPORTS",
        "REFUTES": "REFUTES",
        "NOT ENOUGH INFO": "NOT_ENOUGH_INFO",
        "NOTENOUGHINFO": "NOT_ENOUGH_INFO",
    })

    # Paths — point these at your real data before running data_prep.py
    raw_data_dir: str = "./raw_data"          # expects fever/*.jsonl and liar/*.tsv (or .csv)
    processed_dir: str = "./data"
    output_dir: str = "./model_out"

    # Training
    batch_size: int = 16
    grad_accum_steps: int = 2                 # effective batch 32
    learning_rate: float = 1e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    fp16: bool = False
    bf16: bool = True
    seed: int = 42

    # Chunked training: evaluate every N steps instead of once per epoch, so
    # macro-F1 problems surface early and the best checkpoint (not just the
    # last one) gets kept.
    eval_steps: int = 30

    # Class weighting (inverse frequency, like the fallacy classifier)
    use_class_weights: bool = True

    # Confidence gate for the ensemble hookup (mirrors FallacyAgent's 0.65 pattern)
    confidence_threshold: float = 0.70

    def __post_init__(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

CFG = Config()
