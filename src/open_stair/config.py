# SPDX-License-Identifier: Apache-2.0
"""Published STAIR settings; unspecified engineering choices are explicit."""

import math
from dataclasses import dataclass, fields

PAPER_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
PAPER_QA_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 200
    patience: int = 20
    max_input_length: int = 14000
    max_output_length: int = 64
    lora_rank: int = 16
    lora_alpha: int = 32
    # The paper does not specify these engineering choices.
    learning_rate: float = 2e-4
    batch_size: int = 1
    accumulation_steps: int = 1
    lora_dropout: float = 0.05
    seed: int = 42
    device: str = "auto"
    target_modules: str = "all-linear"
    dtype: str = "auto"
    gradient_checkpointing: bool = True

    def __post_init__(self):
        integer_fields = (
            "epochs",
            "patience",
            "max_input_length",
            "max_output_length",
            "lora_rank",
            "lora_alpha",
            "batch_size",
            "accumulation_steps",
            "seed",
        )
        for name in integer_fields:
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")
        sizes = (
            self.epochs,
            self.patience,
            self.max_input_length,
            self.max_output_length,
            self.batch_size,
            self.accumulation_steps,
            self.lora_alpha,
        )
        if min(sizes) < 1 or self.lora_rank < 0:
            raise ValueError("Training sizes must be positive; lora_rank must be >= 0")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be positive and finite")
        if (
            isinstance(self.lora_dropout, bool)
            or not isinstance(self.lora_dropout, (int, float))
            or not 0 <= self.lora_dropout < 1
        ):
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.dtype not in {"auto", "float32", "bfloat16"}:
            raise ValueError("dtype must be auto, float32, or bfloat16")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps")
        if not isinstance(self.gradient_checkpointing, bool):
            raise ValueError("gradient_checkpointing must be a bool")
        if not isinstance(self.target_modules, str) or not self.target_modules.strip():
            raise ValueError("target_modules must be a nonempty PEFT module selector")

    @classmethod
    def from_dict(cls, value: dict) -> "TrainConfig":
        """Reject misspelled settings before a model is downloaded."""
        if not isinstance(value, dict):
            raise ValueError("Training config must be an object")
        unknown = set(value) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown training settings: {', '.join(sorted(unknown))}")
        return cls(**value)
