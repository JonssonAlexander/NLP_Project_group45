"""
Training utilities for recurrent models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict

import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from .dataloaders import Batch, DataLoaders, build_dataloaders, build_label_mapping
from .dataset_pipeline import TokenisedDatasets
from .embeddings import EmbeddingLoaderResult, Vocabulary
from .models.rnn import PoolingType, RNNClassifier
from .evaluation import evaluate_model


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    best_epoch: int = 0


@dataclass(frozen=True)
class RNNExperimentConfig:
    """
    Bundle of hyperparameters for Part 2 experiments.
    """

    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 0.0
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.3
    pooling: PoolingType = "last_hidden"
    optimizer: str = "adam"
    early_stopping_patience: int | None = None


def train_rnn_model(
    config: RNNExperimentConfig,
    splits: TokenisedDatasets,
    vocabulary: Vocabulary,
    embedding_result: EmbeddingLoaderResult,
    device: torch.device | None = None,
) -> tuple[TrainingHistory, RNNClassifier, Dict[str, int], DataLoaders]:
    """
    Train an RNN classifier and return history, label mapping, and best state dict.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_to_index = build_label_mapping(splits.train)
    dataloaders = build_dataloaders(
        splits=splits,
        vocabulary=vocabulary,
        label_to_index=label_to_index,
        batch_size=config.batch_size,
    )

    num_classes = len(label_to_index)
    embedding_tensor = torch.tensor(embedding_result.matrix, dtype=torch.float32)

    model = RNNClassifier(
        embedding_matrix=embedding_tensor,
        num_classes=num_classes,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        pooling=config.pooling,
        trainable_embeddings=embedding_result.trainable,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(
        config=config,
        model_parameters=model.parameters(),
    )

    history = TrainingHistory()
    best_val_accuracy = 0.0
    best_state: dict | None = None
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = _run_epoch(
            model=model,
            dataloader=dataloaders.train,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=config.grad_clip,
        )
        val_loss, val_acc = _evaluate(
            model=model,
            dataloader=dataloaders.validation,
            criterion=criterion,
            device=device,
        )

        history.train_loss.append(train_loss)
        history.train_accuracy.append(train_acc)
        history.val_loss.append(val_loss)
        history.val_accuracy.append(val_acc)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            history.best_epoch = epoch
            best_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            config.early_stopping_patience is not None
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            break

    if best_state is None:
        best_state = model.state_dict()

    model.load_state_dict(best_state)

    return history, model, label_to_index, dataloaders


def _build_optimizer(
    config: RNNExperimentConfig,
    model_parameters,
) -> torch.optim.Optimizer:
    if config.optimizer.lower() == "adam":
        return Adam(
            model_parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if config.optimizer.lower() == "sgd":
        return SGD(
            model_parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def _run_epoch(
    model: RNNClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)
        optimizer.zero_grad()
        logits = model(batch.token_indices, batch.lengths)
        loss = criterion(logits, batch.labels)
        loss.backward()
        if grad_clip and grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * batch.labels.size(0)
        total_correct += _count_correct(logits, batch.labels)
        total_examples += batch.labels.size(0)

    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    return avg_loss, accuracy


def _evaluate(
    model: RNNClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)
            logits = model(batch.token_indices, batch.lengths)
            loss = criterion(logits, batch.labels)

            total_loss += loss.item() * batch.labels.size(0)
            total_correct += _count_correct(logits, batch.labels)
            total_examples += batch.labels.size(0)

    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    return avg_loss, accuracy


def _evaluate_generic(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)
            logits = _forward_with_lengths(model, batch.token_indices, batch.lengths)
            loss = criterion(logits, batch.labels)
            total_loss += loss.item() * batch.labels.size(0)
            total_correct += _count_correct(logits, batch.labels)
            total_examples += batch.labels.size(0)

    return total_loss / total_examples, total_correct / total_examples


def _forward_with_lengths(
    model: nn.Module,
    tokens: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    try:
        return model(tokens, lengths)
    except TypeError:
        return model(tokens)


def build_dataloaders_and_vocab(
    config: RNNExperimentConfig,
    splits: TokenisedDatasets,
    vocabulary: Vocabulary,
) -> tuple[DataLoaders, Dict[str, int]]:
    """
    Convenience wrapper to obtain dataloaders and the label mapping.
    """

    label_to_index = build_label_mapping(splits.train)
    loaders = build_dataloaders(
        splits=splits,
        vocabulary=vocabulary,
        label_to_index=label_to_index,
        batch_size=config.batch_size,
    )
    return loaders, label_to_index


def train_generic_model(
    model: nn.Module,
    loaders: DataLoaders,
    config: RNNExperimentConfig,
    device: torch.device | None = None,
) -> tuple[TrainingHistory, nn.Module]:
    """
    Train an arbitrary sequence model using the shared dataloaders.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(config, model.parameters())

    history = TrainingHistory()
    best_val_acc = float("-inf")
    best_state: dict | None = None
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for batch in loaders.train:
            tokens = batch.token_indices.to(device)
            lengths = batch.lengths.to(device)
            labels = batch.labels.to(device)

            optimizer.zero_grad()
            logits = _forward_with_lengths(model, tokens, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            if config.grad_clip and config.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += _count_correct(logits, labels)
            total_examples += labels.size(0)

        history.train_loss.append(total_loss / total_examples)
        history.train_accuracy.append(total_correct / total_examples)

        val_loss, val_acc = _evaluate_generic(model, loaders.validation, criterion, device)
        history.val_loss.append(val_loss)
        history.val_accuracy.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history.best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            config.early_stopping_patience is not None
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history, model


def run_model_experiment(
    name: str,
    model_builder: Callable[[], nn.Module],
    config: RNNExperimentConfig,
    loaders: DataLoaders,
    device: torch.device | None = None,
) -> tuple[TrainingHistory, nn.Module, object]:
    """
    Train and evaluate a model returning history, tuned model, and test accuracy.
    """

    model = model_builder()
    history, tuned_model = train_generic_model(model, loaders, config, device=device)
    eval_result = evaluate_model(tuned_model, loaders.test, device=device)
    return history, tuned_model, eval_result


def summarise_run(
    name: str,
    history: TrainingHistory,
    test_accuracy: float,
) -> dict[str, float]:
    """
    Compact dictionary summarising an experiment outcome.
    """

    if history.best_epoch and history.best_epoch <= len(history.val_accuracy):
        best_val = history.val_accuracy[history.best_epoch - 1]
    elif history.val_accuracy:
        best_val = history.val_accuracy[-1]
    else:
        best_val = float("nan")

    return {
        "name": name,
        "best_epoch": history.best_epoch,
        "val_accuracy": best_val,
        "test_accuracy": test_accuracy,
    }


def _move_batch_to_device(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        token_indices=batch.token_indices.to(device),
        lengths=batch.lengths.to(device),
        labels=batch.labels.to(device),
    )


def _count_correct(logits: torch.Tensor, labels: torch.Tensor) -> int:
    predictions = torch.argmax(logits, dim=1)
    return int((predictions == labels).sum().item())
