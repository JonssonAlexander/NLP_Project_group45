"""
ABBR improved RCNN using oversampling
"""

import torch
from dataclasses import dataclass
from typing import List, Tuple
from src.models.rcnn import RCNNTextClassifier
from src.training import build_dataloaders_and_vocab, train_generic_model
from src.evaluation import evaluate_model


@dataclass
class CoarseSplits:
    train: List[Tuple[str, List[str]]]
    validation: List[Tuple[str, List[str]]]
    test: List[Tuple[str, List[str]]]


def _to_coarse_label(label: str) -> str:
    return label.split(":")[0] if ":" in label else label


def coarsen_dataset(ds: List[Tuple[str, List[str]]]) -> List[Tuple[str, List[str]]]:
    return [(_to_coarse_label(lbl), toks) for lbl, toks in ds]


def make_coarse_splits(splits) -> CoarseSplits:
    return CoarseSplits(
        train=coarsen_dataset(splits.train),
        validation=coarsen_dataset(splits.validation),
        test=coarsen_dataset(splits.test),
    )


def oversample_label(examples: List[Tuple[str, List[str]]], target: str, factor: int = 3):
    base = [(y, x) for (y, x) in examples]
    boost = [(y, x) for (y, x) in examples if y == target] * max(0, factor - 1)
    return base + boost


def build_dataloaders(config, splits, vocab_report):
    loaders, label_to_index = build_dataloaders_and_vocab(
        config=config,
        splits=splits,
        vocabulary=vocab_report.vocabulary,
    )
    return loaders, label_to_index


def train_abbr(loaders, config, device, embedding_tensor):
    if hasattr(loaders, "label_to_index"):
        label_map = loaders.label_to_index
    elif hasattr(loaders, "train") and hasattr(loaders.train, "dataset") and hasattr(loaders.train.dataset, "label_to_index"):
        label_map = loaders.train.dataset.label_to_index

    num_classes = len(label_map)

    model = RCNNTextClassifier(
        embedding_matrix=embedding_tensor.clone(),
        num_classes=num_classes,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        padding_idx=0,
        multisample_dropout=4,
    ).to(device)

    history, tuned = train_generic_model(model, loaders, config, device=device)
    eval_result = evaluate_model(tuned, loaders.test, device=device)
    return history, tuned, eval_result