"""
Plotting helpers to keep notebooks focused on storytelling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .embeddings import Vocabulary
from .training import TrainingHistory


def _prepare_tokens(tokens: Sequence[str] | Sequence[tuple[str, int]]) -> list[str]:
    sequence: list[str] = []
    for entry in tokens:
        if isinstance(entry, tuple):
            sequence.append(entry[0])
        else:
            sequence.append(entry)
    return sequence


def project_embeddings_2d(
    tokens_by_label: Mapping[str, Sequence[str] | Sequence[tuple[str, int]]],
    vocabulary: Vocabulary,
    embedding_matrix,
    method: str = "pca",
    random_state: int = 42,
    tsne_perplexity: float = 30.0,
):
    """
    Reduce token embeddings to 2D using PCA or t-SNE.
    """

    tokens: list[str] = []
    labels: list[str] = []
    vectors = []

    for label, entries in tokens_by_label.items():
        for token in _prepare_tokens(entries):
            index = vocabulary.lookup(token, default=-1)
            if index < 0:
                continue
            tokens.append(token)
            labels.append(label)
            vectors.append(embedding_matrix[index])

    if not vectors:
        raise ValueError("No embeddings available for the provided tokens.")

    vectors_array = np.vstack(vectors)

    if method.lower() == "tsne":
        effective_perplexity = min(tsne_perplexity, max(5.0, len(vectors_array) - 1))
        reducer = TSNE(
            n_components=2,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
            perplexity=effective_perplexity,
        )
    else:
        reducer = PCA(n_components=2, random_state=random_state)

    reduced = reducer.fit_transform(vectors_array)

    frame = pd.DataFrame(reduced, columns=["x", "y"])
    frame["token"] = tokens
    frame["label"] = labels
    return frame


def plot_embedding_scatter(
    frame: pd.DataFrame,
    title: str,
    output_path: Path | None = None,
    annotate_tokens: bool = True,
    legend_location: str = "right",
):
    """
    Scatter plot helper for 2D embedding projections.
    """

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))

    sns.scatterplot(
        data=frame,
        x="x",
        y="y",
        hue="label",
        palette="tab10",
        s=80,
        ax=ax,
    )

    if annotate_tokens:
        for _, row in frame.iterrows():
            ax.text(row["x"] + 0.02, row["y"] + 0.02, row["token"], fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    if legend_location == "bottom":
        ax.legend(
            title="Topic",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=3,
        )
    elif legend_location == "none":
        ax.legend_.remove()
    else:
        ax.legend(title="Topic", bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)

    return fig, ax


def plot_top_tokens_projection(
    tokens_by_label: Mapping[str, Sequence[str] | Sequence[tuple[str, int]]],
    vocabulary: Vocabulary,
    embedding_matrix,
    method: str = "pca",
    title: str | None = None,
    output_path: Path | None = None,
    random_state: int = 42,
    annotate_tokens: bool = True,
    legend_location: str = "right",
    tsne_perplexity: float = 30.0,
):
    """
    Convenience wrapper: project embeddings and produce scatter plot.
    """

    frame = project_embeddings_2d(
        tokens_by_label=tokens_by_label,
        vocabulary=vocabulary,
        embedding_matrix=embedding_matrix,
        method=method,
        random_state=random_state,
        tsne_perplexity=tsne_perplexity,
    )

    fig, ax = plot_embedding_scatter(
        frame=frame,
        title=title or f"Embedding projection ({method.upper()})",
        output_path=output_path,
        annotate_tokens=annotate_tokens,
        legend_location=legend_location,
    )

    return frame, fig, ax


def plot_training_curves(
    history: TrainingHistory,
    title: str = "Training dynamics",
    output_path: Path | None = None,
):
    """
    Plot training loss and validation accuracy curves.
    """

    epochs = range(1, len(history.train_loss) + 1)
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history.train_loss, label="Train loss")
    axes[0].plot(epochs, history.val_loss, label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history.train_accuracy, label="Train acc")
    axes[1].plot(epochs, history.val_accuracy, label="Val acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)

    return fig, axes
