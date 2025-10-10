"""
Embedding utilities: vocabulary construction, pretrained vector loading,
and OOV accounting shared across notebooks.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Vocabulary:
    """
    Simple vocabulary container with frequency metadata.
    """

    token_to_index: Mapping[str, int]
    index_to_token: Sequence[str]
    frequencies: Mapping[str, int]

    def __len__(self) -> int:
        return len(self.index_to_token)

    def lookup(self, token: str, default: int | None = None) -> int:
        if default is None:
            return self.token_to_index[token]
        return self.token_to_index.get(token, default)


def build_vocabulary(
    tokenised_sequences: Iterable[Sequence[str]],
    min_freq: int = 1,
    specials: Sequence[str] = ("<pad>", "<unk>"),
) -> Vocabulary:
    """
    Construct a vocabulary from tokenised sequences.
    """

    counter: Counter[str] = Counter()
    for tokens in tokenised_sequences:
        counter.update(tokens)

    token_to_index: dict[str, int] = {}
    index_to_token: list[str] = []

    for special in specials:
        if special in token_to_index:
            continue
        token_to_index[special] = len(index_to_token)
        index_to_token.append(special)

    for token, frequency in counter.most_common():
        if frequency < min_freq:
            continue
        if token in token_to_index:
            continue
        token_to_index[token] = len(index_to_token)
        index_to_token.append(token)

    return Vocabulary(token_to_index=token_to_index, index_to_token=index_to_token, frequencies=counter)


@dataclass(frozen=True)
class EmbeddingConfig:
    """
    Configuration for loading pretrained embeddings.
    """

    file_path: Path
    embedding_dim: int
    trainable: bool = True
    random_seed: int = 7


@dataclass(frozen=True)
class EmbeddingLoaderResult:
    """
    Output structure for pretrained embedding loading.
    """

    matrix: np.ndarray
    oov_tokens: set[str]
    trainable: bool
    mitigated_tokens: Mapping[str, np.ndarray]


def load_glove_embeddings(
    config: EmbeddingConfig,
    vocabulary: Vocabulary,
) -> EmbeddingLoaderResult:
    """
    Load GloVe-style embeddings (plain text, token followed by floats per line).
    """

    if not config.file_path.exists():
        raise FileNotFoundError(f"GloVe file not found: {config.file_path}")

    rng = np.random.default_rng(config.random_seed)
    embedding_matrix = rng.normal(
        loc=0.0,
        scale=0.1,
        size=(len(vocabulary), config.embedding_dim),
    ).astype(np.float32)

    found_tokens: set[str] = set()
    with config.file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != config.embedding_dim + 1:
                continue
            token = parts[0]
            maybe_index = vocabulary.token_to_index.get(token)
            if maybe_index is None:
                continue
            embedding_matrix[maybe_index] = np.asarray(parts[1:], dtype=np.float32)
            found_tokens.add(token)

    oov_tokens = {token for token in vocabulary.token_to_index if token not in found_tokens}
    mitigated = mitigate_oov_embeddings(
        embedding_matrix,
        vocabulary,
        oov_tokens,
    )
    return EmbeddingLoaderResult(
        matrix=embedding_matrix,
        oov_tokens=oov_tokens,
        trainable=config.trainable,
        mitigated_tokens=mitigated,
    )
# above - instead of returning all three elements separately, I thought that it would be better to create an "organised object"


def load_torchtext_glove(
    vocabulary: Vocabulary,
    name: str = "6B",
    dim: int = 100,
    trainable: bool = True,
    random_seed: int = 7,
) -> EmbeddingLoaderResult:
    """
    Load GloVe embeddings via torchtext's downloader.

    Torchtext caches the vectors locally (under `~/.vector_cache`) after the
    first download. Vocabulary items without pretrained vectors remain
    randomly initialised so they can be fine-tuned during training.
    """

    try:
        from torchtext.vocab import GloVe
    except ImportError as exc:  # pragma: no cover - dependency availability
        raise RuntimeError(
            "torchtext must be installed to load GloVe embeddings via torchtext.vocab.GloVe."
        ) from exc

    glove = GloVe(name=name, dim=dim)

    rng = np.random.default_rng(random_seed)
    embedding_matrix = rng.normal(
        loc=0.0,
        scale=0.1,
        size=(len(vocabulary), dim),
    ).astype(np.float32)

    found_tokens: set[str] = set()
    for token, index in vocabulary.token_to_index.items():
        stoi_index = glove.stoi.get(token)
        if stoi_index is None:
            continue
        embedding_matrix[index] = glove.vectors[stoi_index].numpy()
        found_tokens.add(token)

    oov_tokens = {token for token in vocabulary.token_to_index if token not in found_tokens}
    mitigated = mitigate_oov_embeddings(
        embedding_matrix,
        vocabulary,
        oov_tokens,
    )
    return EmbeddingLoaderResult(
        matrix=embedding_matrix,
        oov_tokens=oov_tokens,
        trainable=trainable,
        mitigated_tokens=mitigated,
    )


def mitigate_oov_embeddings(
    embedding_matrix: np.ndarray,
    vocabulary: Vocabulary,
    oov_tokens: set[str],
) -> Mapping[str, np.ndarray]:
    """
    Naively mitigate OOV tokens by assigning them the global mean embedding.

    This simple strategy gives OOV tokens a deterministic starting point rather
    than random noise, while keeping the vectors trainable so the model can adapt
    them during learning.
    """

    mitigated: dict[str, np.ndarray] = {}

    if not oov_tokens:
        return mitigated

    in_vocab_indices = [
        idx
        for token, idx in vocabulary.token_to_index.items()
        if token not in {"<pad>", "<unk>"} and token not in oov_tokens
    ]

    if not in_vocab_indices:
        return mitigated

    mean_vector = embedding_matrix[in_vocab_indices].mean(axis=0)

    for token in sorted(oov_tokens):
        token_index = vocabulary.token_to_index.get(token)
        if token_index is None:
            continue
        embedding_matrix[token_index] = mean_vector.copy()
        mitigated[token] = mean_vector.copy()

    return mitigated


def compute_oov_by_label(
    labeled_sequences: Iterable[tuple[str, Sequence[str]]],
    oov_tokens: set[str],
) -> Mapping[str, int]:
    """
    Count OOV tokens per label category.
    """

    counts: dict[str, int] = defaultdict(int)
    for label, tokens in labeled_sequences:
        counts[label] += sum(1 for token in tokens if token in oov_tokens)
    return counts
