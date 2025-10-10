"""
Evaluation utilities for the RNN experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from .dataloaders import Batch


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    accuracy: float
    predictions: list[int]
    targets: list[int]


def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    device: torch.device | None = None,
) -> EvaluationResult:
    """
    Compute loss and accuracy on a dataloader, returning predictions/targets for further analysis.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    predictions: list[int] = []
    targets: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)
            logits = model(batch.token_indices, batch.lengths)
            loss = criterion(logits, batch.labels)

            total_loss += loss.item() * batch.labels.size(0)
            pred = torch.argmax(logits, dim=1)
            total_correct += int((pred == batch.labels).sum().item())
            total_examples += batch.labels.size(0)

            predictions.extend(pred.cpu().tolist())
            targets.extend(batch.labels.cpu().tolist())

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return EvaluationResult(
        loss=average_loss,
        accuracy=accuracy,
        predictions=predictions,
        targets=targets,
    )


def topic_accuracy_table(
    result: EvaluationResult,
    index_to_label: Mapping[int, str],
) -> pd.DataFrame:
    """
    Produce a dataframe with per-topic accuracy and support.
    """

    counts = {
        label: {"correct": 0, "total": 0}
        for label in index_to_label.values()
    }

    for pred, target in zip(result.predictions, result.targets):
        label = index_to_label[target]
        counts[label]["total"] += 1
        if pred == target:
            counts[label]["correct"] += 1

    rows = []
    for label, stats in counts.items():
        total = stats["total"]
        correct = stats["correct"]
        accuracy = correct / total if total > 0 else 0.0
        rows.append({"label": label, "accuracy": accuracy, "support": total})

    return pd.DataFrame(rows).sort_values("label").reset_index(drop=True)


def classification_report_table(
    result: EvaluationResult,
    index_to_label: Mapping[int, str],
) -> pd.DataFrame:
    """
    Generate a precision/recall/F1 table using sklearn.
    """

    labels = sorted(index_to_label.keys())
    report = classification_report(
        result.targets,
        result.predictions,
        labels=labels,
        target_names=[index_to_label[idx] for idx in labels],
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose()


def confusion_matrix_frame(
    result: EvaluationResult,
    index_to_label: Mapping[int, str],
) -> pd.DataFrame:
    """
    Produce a confusion matrix as a pandas DataFrame.
    """

    labels = sorted(index_to_label.keys())
    matrix = confusion_matrix(result.targets, result.predictions, labels=labels)
    label_names = [index_to_label[idx] for idx in labels]

    return pd.DataFrame(matrix, index=label_names, columns=label_names)


def _move_batch_to_device(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        token_indices=batch.token_indices.to(device),
        lengths=batch.lengths.to(device),
        labels=batch.labels.to(device),
    )
