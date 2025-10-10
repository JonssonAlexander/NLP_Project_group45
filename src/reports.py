"""
Reporting helpers to keep notebooks lightweight and focused on narrative.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .embeddings import EmbeddingLoaderResult, Vocabulary, build_vocabulary


@dataclass(frozen=True)
class VocabularyReport:
    """
    Aggregated statistics for a trained vocabulary.
    """

    vocabulary: Vocabulary
    token_counts: Counter[str]
    document_count: int
    min_freq: int
    specials: tuple[str, ...]

    @property
    def total_tokens(self) -> int:
        return sum(self.token_counts.values())

    @property
    def unique_tokens(self) -> int:
        return len(self.token_counts)

    @property
    def vocabulary_size(self) -> int:
        return len(self.vocabulary)

    def summary(self) -> dict[str, int]:
        return {
            "documents": self.document_count,
            "total_tokens": self.total_tokens,
            "unique_tokens": self.unique_tokens,
            "vocabulary_size": self.vocabulary_size,
            "min_frequency_threshold": self.min_freq,
        }

    def most_common(self, n: int = 20) -> list[tuple[str, int]]:
        return self.token_counts.most_common(n)


def build_vocabulary_report(
    tokenised_dataset: Sequence[tuple[str, Sequence[str]]],
    min_freq: int,
    specials: Sequence[str],
) -> VocabularyReport:
    """
    Produce a reusable vocabulary report for downstream analysis.
    """

    token_counts: Counter[str] = Counter()
    for _, tokens in tokenised_dataset:
        token_counts.update(tokens)

    vocabulary = build_vocabulary(
        (tokens for _, tokens in tokenised_dataset),
        min_freq=min_freq,
        specials=specials,
    )

    return VocabularyReport(
        vocabulary=vocabulary,
        token_counts=token_counts,
        document_count=len(tokenised_dataset),
        min_freq=min_freq,
        specials=tuple(specials),
    )


@dataclass(frozen=True)
class OOVReport:
    """
    OOV summary including both unique word counts and total occurrences.
    """

    embedding_result: EmbeddingLoaderResult
    unique_oov_tokens: frozenset[str]
    token_occurrence_counts: Mapping[str, int]
    token_count_by_label: Mapping[str, int]
    unique_count_by_label: Mapping[str, int]
    mitigated_tokens: Mapping[str, np.ndarray] | None = None

    @property
    def total_unique_oov(self) -> int:
        return len(self.unique_oov_tokens)

    @property
    def total_oov_occurrences(self) -> int:
        return sum(self.token_occurrence_counts.values())

    @property
    def mitigated_count(self) -> int:
        return len(self.mitigated_tokens or {})

    def summary(self) -> dict[str, int]:
        return {
            "unique_oov_words": self.total_unique_oov,
            "oov_token_occurrences": self.total_oov_occurrences,
            "mitigated_vectors": len(self.mitigated_tokens or {}),
        }

    def per_label_summary(self) -> list[dict[str, int]]:
        labels = sorted(self.token_count_by_label.keys())
        return [
            {
                "label": label,
                "unique_oov_words": self.unique_count_by_label.get(label, 0),
                "oov_token_occurrences": self.token_count_by_label.get(label, 0),
            }
            for label in labels
        ]

    def top_oov_tokens(self, n: int = 20) -> list[tuple[str, int]]:
        return sorted(
            self.token_occurrence_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:n]


def compute_oov_report(
    tokenised_dataset: Sequence[tuple[str, Sequence[str]]],
    special_tokens: Sequence[str],
    embedding_result: EmbeddingLoaderResult,
) -> OOVReport:
    """
    Build OOV statistics for quick inspection inside notebooks.
    """

    specials = set(special_tokens)
    oov_tokens = {token for token in embedding_result.oov_tokens if token not in specials}

    token_occurrence_counter: Counter[str] = Counter()
    occurrences_by_label: defaultdict[str, int] = defaultdict(int)
    unique_by_label: defaultdict[str, set[str]] = defaultdict(set)

    for label, tokens in tokenised_dataset:
        label_unique: set[str] = set()
        for token in tokens:
            if token not in oov_tokens:
                continue
            token_occurrence_counter[token] += 1
            occurrences_by_label[label] += 1
            label_unique.add(token)
        unique_by_label[label].update(label_unique)

    unique_count_by_label = {label: len(tokens) for label, tokens in unique_by_label.items()}

    return OOVReport(
        embedding_result=embedding_result,
        unique_oov_tokens=frozenset(token_occurrence_counter.keys()),
        token_occurrence_counts=dict(token_occurrence_counter),
        token_count_by_label=dict(occurrences_by_label),
        unique_count_by_label=unique_count_by_label,
        mitigated_tokens=embedding_result.mitigated_tokens,
    )


def top_tokens_by_label(
    tokenised_dataset: Sequence[tuple[str, Sequence[str]]],
    top_k: int = 20,
    stopwords: set[str] | None = None,
) -> Mapping[str, list[tuple[str, int]]]:
    """
    Return top-k frequent tokens per label, with optional stopword filtering.
    """

    counters: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for label, tokens in tokenised_dataset:
        for token in tokens:
            if stopwords and token in stopwords:
                continue
            counters[label][token] += 1

    return {label: counter.most_common(top_k) for label, counter in counters.items()}
