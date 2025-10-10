"""
Configuration loading helpers so notebooks can stay declarative.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from .preprocessing import TokenisationConfig

if TYPE_CHECKING:  # pragma: no cover
    from .embeddings import EmbeddingConfig


@dataclass(frozen=True)
class DataConfig:
    """
    Structured representation of data-related settings.
    """

    dataset_root: Path
    train_filename: str
    test_filename: str
    train_split_ratio: float
    shuffle_seed: int
    tokenisation: TokenisationConfig
    vocabulary_min_freq: int
    vocabulary_specials: tuple[str, ...]


def load_data_config(path: Path) -> DataConfig:
    """
    Load the YAML configuration into a strongly typed structure.
    """

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    token_payload = payload.get("tokenisation", {})
    token_config = TokenisationConfig(
        use_spacy=token_payload.get("use_spacy", True),
        language_model=token_payload.get("language_model", "en_core_web_sm"),
        lower=token_payload.get("lower", True),
    )

    vocab_payload = payload.get("vocabulary", {})
    return DataConfig(
        dataset_root=Path(payload["dataset_root"]),
        train_filename=payload["train_filename"],
        test_filename=payload["test_filename"],
        train_split_ratio=float(payload["train_split_ratio"]),
        shuffle_seed=int(payload["shuffle_seed"]),
        tokenisation=token_config,
        vocabulary_min_freq=int(vocab_payload.get("min_freq", 1)),
        vocabulary_specials=tuple(vocab_payload.get("specials", ("<pad>", "<unk>"))),
    )


def load_embedding_config(path: Path) -> "EmbeddingConfig":
    """
    Load embedding configuration and return an EmbeddingConfig instance.
    """

    from .embeddings import EmbeddingConfig  # Local import to avoid cycles

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    file_path = Path(payload["file_path"])
    if not file_path.is_absolute():
        file_path = (path.parent / file_path).resolve()

    return EmbeddingConfig(
        file_path=file_path,
        embedding_dim=int(payload["embedding_dim"]),
        trainable=bool(payload.get("trainable", True)),
        random_seed=int(payload.get("random_seed", 7)),
    )
