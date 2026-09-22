"""Validated fresh-Qwen or utility-checkpoint training configuration."""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    suite: str
    base_model: str
    base_revision: str | None = None
    init_checkpoint: str | None = None
    init_checkpoint_revision: str | None = None
    reviewed_proposals: str | None = None
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    dtype: Literal["auto", "fp32", "bf16"] = "auto"
    attention_mode: Literal["auto", "flash", "reference"] = "auto"
    decode_check_interval: int = Field(default=8, ge=1, le=32)
    lora_rank: int = Field(default=16, ge=1)
    head_dim: int = Field(default=128, ge=1)
    omission_heads: int = Field(default=4, ge=1)
    omission_layers: int = Field(default=2, ge=1)
    epochs: int = Field(default=3, ge=1)
    learning_rate: float = Field(default=0.0001, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    accumulation_steps: int = Field(default=4, ge=1)
    max_steps: int | None = Field(default=None, ge=1)
    max_state_tokens: int = Field(default=384, ge=8)
    max_length: int = Field(default=1024, ge=32)
    max_packed_tokens: int = Field(default=4096, ge=32)
    max_questions_per_pass: int = Field(default=8, ge=1, le=64)
    max_cache_tokens: int = Field(default=4096, ge=32)
    max_decode_batch_tokens: int = Field(default=16384, ge=32)
    branch_bucket_width: int = Field(default=0, ge=0, le=256)
    admission_margin: float = Field(default=1.3862943611198906, gt=0)
    branch_batch_size: int = Field(default=16, ge=1, le=256)
    score_weight: float = Field(default=1.0, gt=0)
    omission_weight: float = Field(default=1.0, gt=0)
    generation_weight: float = Field(default=1.0, gt=0)
    acceptance_threshold: float = Field(default=0.8, gt=0.5, lt=1)
    seed: int = 42
    gradient_checkpointing: bool = False

    @model_validator(mode="after")
    def validate_context(self):
        if self.head_dim % self.omission_heads:
            raise ValueError("head_dim must be divisible by omission_heads")
        if self.max_state_tokens >= self.max_length:
            raise ValueError("max_state_tokens must leave room for questions and actions")
        if self.max_packed_tokens < self.max_length:
            raise ValueError("max_packed_tokens must be at least max_length")
        if self.init_checkpoint_revision and not self.init_checkpoint:
            raise ValueError("init_checkpoint_revision requires init_checkpoint")
        if not Path(self.base_model).is_dir() and not self.base_revision:
            raise ValueError("pin base_revision for a Hub model")
        return self


def resolve_checkpoint_reference(value, directory):
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else directory / raw
    if raw.is_absolute() or value.startswith(("./", "../", "~")) or candidate.exists():
        return str(candidate.resolve())
    return value


def load_config(path):
    path = Path(path).resolve()
    raw = tomllib.loads(path.read_text())
    for field in ("base_model", "init_checkpoint"):
        if raw.get(field):
            raw[field] = resolve_checkpoint_reference(raw[field], path.parent)
    if raw.get("reviewed_proposals"):
        raw["reviewed_proposals"] = str((path.parent / raw["reviewed_proposals"]).resolve())
    config = TrainingConfig.model_validate(raw)
    return config, (path.parent / config.suite).resolve()


def device_for(requested):
    import torch

    if requested == "auto":
        return (
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    return requested
