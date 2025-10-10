"""
Utilities that stitch together config, IO, and preprocessing steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .config import DataConfig
from .data_io import (
    TrecDatasetPaths,
    ensure_dataset_local,
    read_labeled_questions,
    shuffle_examples,
    split_train_validation,
)
from .preprocessing import Tokeniser, build_simple_tokeniser, prepare_tokenised_dataset


@dataclass(frozen=True)
class TokenisedDatasets:
    """
    Container for tokenised splits ready for vocabulary building.
    """

    train: Sequence[tuple[str, Sequence[str]]]
    validation: Sequence[tuple[str, Sequence[str]]]
    test: Sequence[tuple[str, Sequence[str]]]


def _materialise(path_iterator: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    return list(path_iterator)


def prepare_tokenised_splits(config: DataConfig) -> TokenisedDatasets:
    """
    Ensure the dataset exists locally and return tokenised train/val/test splits.
    """

    paths = TrecDatasetPaths(
        root=config.dataset_root,
        train_filename=config.train_filename,
        test_filename=config.test_filename,
    )
    ensure_dataset_local(paths)

    train_examples = _materialise(read_labeled_questions(paths.train_path))
    test_examples = _materialise(read_labeled_questions(paths.test_path))

    shuffled = shuffle_examples(train_examples, config.shuffle_seed)
    train_split, validation_split = split_train_validation(shuffled, config.train_split_ratio)

    tokeniser: Tokeniser = build_simple_tokeniser(config.tokenisation)
    tokenised_train = prepare_tokenised_dataset(train_split, tokeniser)
    tokenised_validation = prepare_tokenised_dataset(validation_split, tokeniser)
    tokenised_test = prepare_tokenised_dataset(test_examples, tokeniser)

    return TokenisedDatasets(
        train=tokenised_train,
        validation=tokenised_validation,
        test=tokenised_test,
    )
