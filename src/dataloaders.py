"""
Utilities for building PyTorch DataLoaders from tokenised datasets.

These helpers keep batching logic out of the notebooks so training loops
can stay concise and configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .embeddings import Vocabulary
from .dataset_pipeline import TokenisedDatasets


@dataclass(frozen=True)
class EncodedExample:
    """
    Numeric representation of a tokenised example.
    """

    token_indices: Sequence[int]
    length: int
    label_index: int


@dataclass(frozen=True)
class Batch:
    """
    Collated batch ready for an RNN model.
    """

    token_indices: torch.Tensor  # shape: (batch, seq_len)
    lengths: torch.Tensor  # shape: (batch,)
    labels: torch.Tensor  # shape: (batch,)


def build_label_mapping(
    tokenised_dataset: Sequence[tuple[str, Sequence[str]]],
) -> Mapping[str, int]:
    """
    Create a consistent label -> index mapping.
    """

    labels = sorted({label for label, _ in tokenised_dataset})
    return {label: idx for idx, label in enumerate(labels)}


def encode_examples(
    tokenised_dataset: Sequence[tuple[str, Sequence[str]]],
    vocabulary: Vocabulary,
    label_to_index: Mapping[str, int],
    pad_token: str = "<pad>",
    unk_token: str = "<unk>",
) -> List[EncodedExample]:
    """
    Convert a tokenised dataset into integer indices.
    """

    pad_index = vocabulary.token_to_index.get(pad_token, 0)
    unk_index = vocabulary.token_to_index.get(unk_token, pad_index)

    encoded: list[EncodedExample] = []
    for label, tokens in tokenised_dataset:
        indices = [vocabulary.lookup(token, default=unk_index) for token in tokens]
        encoded.append(
            EncodedExample(
                token_indices=indices,
                length=len(indices),
                label_index=label_to_index[label],
            )
        )
    return encoded


class TokenDataset(Dataset):
    """
    Thin wrapper to expose encoded examples via PyTorch Dataset interface.
    """

    def __init__(self, encoded_examples: Sequence[EncodedExample]):
        self._examples = list(encoded_examples)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._examples)

    def __getitem__(self, index: int) -> EncodedExample:
        return self._examples[index]


def _collate_fn(
    batch: Sequence[EncodedExample],
    padding_index: int,
) -> Batch:
    """
    Convert a list of EncodedExample into padded tensors.
    """

    token_tensors = [
        torch.tensor(example.token_indices, dtype=torch.long) for example in batch
    ]
    lengths = torch.tensor([example.length for example in batch], dtype=torch.long)
    labels = torch.tensor([example.label_index for example in batch], dtype=torch.long)

    padded = pad_sequence(token_tensors, batch_first=True, padding_value=padding_index)

    return Batch(token_indices=padded, lengths=lengths, labels=labels)


def create_data_loader(
    encoded_examples: Sequence[EncodedExample],
    vocabulary: Vocabulary,
    batch_size: int,
    shuffle: bool,
    pad_token: str = "<pad>",
    num_workers: int = 0,
) -> DataLoader:
    """
    Build a DataLoader with appropriate padding behaviour.
    """

    dataset = TokenDataset(encoded_examples)
    padding_index = vocabulary.token_to_index.get(pad_token, 0)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: _collate_fn(batch, padding_index=padding_index),
    )


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


def build_dataloaders(
    splits: TokenisedDatasets,
    vocabulary: Vocabulary,
    label_to_index: Mapping[str, int],
    batch_size: int,
    num_workers: int = 0,
) -> DataLoaders:
    """
    Encode datasets and create train/validation/test DataLoaders.
    """

    encoded_train = encode_examples(splits.train, vocabulary, label_to_index)
    encoded_val = encode_examples(splits.validation, vocabulary, label_to_index)
    encoded_test = encode_examples(splits.test, vocabulary, label_to_index)

    return DataLoaders(
        train=create_data_loader(
            encoded_examples=encoded_train,
            vocabulary=vocabulary,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        validation=create_data_loader(
            encoded_examples=encoded_val,
            vocabulary=vocabulary,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        test=create_data_loader(
            encoded_examples=encoded_test,
            vocabulary=vocabulary,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    )
