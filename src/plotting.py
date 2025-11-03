"""
Plotting helpers to keep notebooks focused on storytelling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .embeddings import Vocabulary
from .training import TrainingHistory


def latex_set_size(width: float = 455.24411, fraction: float = 1) -> tuple[float, float]:
    """
    Set figure dimensions to avoid scaling in LaTeX.
    """

    fig_width_pt = width * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5 ** 0.5 - 1) / 2

    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio

    return fig_width_in, fig_height_in


def locale_parameters() -> dict[str, object]:
    """
    Configure matplotlib font defaults for consistent typography.
    """
    rc_overrides = {
        "mathtext.fontset": "stix",
        "font.family": ["STIXGeneral"],
        "font.serif": ["STIXGeneral"],
        "font.sans-serif": ["STIXGeneral"],
        "font.monospace": ["STIXGeneral"],
    }
    plt.rcParams.update(rc_overrides)
    return rc_overrides


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
    sns_style: str = "whitegrid",
    label_palette: Mapping[str, str] | Sequence[str] | None = None,
    legend_label_map: Mapping[str, str] | None = None,
    point_size: float = 80,
):
    """
    Scatter plot helper for 2D embedding projections.
    """

    rc = locale_parameters()
    sns.set(style=sns_style, rc=rc)
    
    figure_size = latex_set_size()

    fig, ax = plt.subplots(figsize=figure_size)

    palette = label_palette if label_palette is not None else "tab10"
    sns.scatterplot(
        data=frame,
        x="x",
        y="y",
        hue="label",
        palette=palette,
        s=point_size,
        ax=ax,
    )

    if annotate_tokens:
        for _, row in frame.iterrows():
            ax.text(row["x"] + 0.02, row["y"] + 0.02, row["token"], fontsize=8)

    handles, labels = ax.get_legend_handles_labels()
    if legend_label_map:
        remapped_handles: list[Any] = []
        remapped_labels: list[str] = []
        seen: set[str] = set()
        for handle, label in zip(handles, labels):
            mapped = legend_label_map.get(label, label)
            if mapped in seen:
                continue
            seen.add(mapped)
            remapped_handles.append(handle)
            remapped_labels.append(mapped)
        handles, labels = remapped_handles, remapped_labels

    ax.set_title(title,fontsize=18)
    ax.set_xlabel("Component 1",fontsize=14)
    ax.set_ylabel("Component 2",fontsize=14)

    if legend_location == "bottom":
        ax.legend(
            handles,
            labels,
            title="Topic",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=3,
        )
    elif legend_location == "none":
        legend_obj = ax.legend_
        if legend_obj is not None:
            legend_obj.remove()
    else:
        ax.legend(
            handles,
            labels,
            title="Topic",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

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
    figure_size: tuple[float, float] | None = None,
    figure_fraction: float | None = None,
    sns_style: str = "whitegrid",
    label_palette: Mapping[str, str] | Sequence[str] | None = None,
    legend_label_map: Mapping[str, str] | None = None,
    point_size: int | None = None
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
        sns_style=sns_style,
        label_palette=label_palette,
        legend_label_map=legend_label_map,
        point_size=point_size,
    )

    return frame, fig, ax


def plot_training_curves(
    history: TrainingHistory,
    title: str = "Training dynamics",
    output_path: Path | None = None,
    sns_style: str = "whitegrid",
    figure_fraction: float | None = None,
):
    """
    Plot training loss and validation accuracy curves.
    """

    epochs = range(1, len(history.train_loss) + 1)
    rc = locale_parameters()
    sns.set(style=sns_style, rc=rc)

    width, height = latex_set_size(fraction=figure_fraction)
    figsize = (width * 2, height)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(epochs, history.train_loss, "-o", label="Training")
    axes[0].plot(epochs, history.val_loss, "-o", label="Validation")
    axes[0].set_xlabel("Epoch",fontsize=14)
    axes[0].set_ylabel("Loss",fontsize=14)
    axes[0].set_title("Loss",fontsize=14)
    axes[0].legend()

    axes[1].plot(epochs, history.train_accuracy, "-o", label="Training")
    axes[1].plot(epochs, history.val_accuracy, "-o", label="Validation")
    axes[1].set_xlabel("Epoch",fontsize=14)
    axes[1].set_ylabel("Accuracy",fontsize=14)
    axes[1].set_title("Accuracy",fontsize=14)
    axes[1].legend()
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    max_epoch = len(history.train_loss)
    axes[0].set_xlim(0.95, max_epoch+0.05)
    axes[1].set_xlim(0.95, max_epoch+0.05)

    def _format_float(value: float, _pos: int) -> str:
        return f"{value:.3f}"

    axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(_format_float))
    axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(_format_float))

    fig.suptitle(title,fontsize=18)
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)

    return fig, axes


def plot_barplot(
    x: Sequence[Any],
    y: Sequence[float | int],
    title: str,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Render a bar plot for the provided x/y values.
    """

    frame = pd.DataFrame({"x": x, "y": y})
    rc = locale_parameters()
    sns.set(style="whitegrid", rc=rc)
    fig, ax = plt.subplots(figsize=latex_set_size(fraction=1))
    sns.barplot(data=frame, x="x", y="y", ax=ax)

    ax.set_title(title,fontsize=18)
    ax.set_xlabel("Label",fontsize=14)
    ax.set_ylabel("Counts",fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    fig.tight_layout()
    return fig, ax


def plot_support_and_f1_by_topic(
    label_rows: pd.DataFrame,
    *,
    accuracy_col: str = "accuracy",
    support_col: str = "support",
    f1_col: str = "f1-score",
    figsize: tuple[float, float] | None = None,
    accuracy_cmap: str = "magma",
    support_title: str = "Support by topic (color = accuracy)",
    f1_title: str = "F1 score by topic (color = accuracy)",
    xlabel: str = "Topic",
    support_ylabel: str = "Support",
    f1_ylabel: str = "F1 score",
    accuracy_label: str = "Accuracy",
    support_output_path: Path | None = None,
    f1_output_path: Path | None = None,
    xtick_rotation: float = 45,
    accuracy_text_offset: float = 5.0,
    f1_text_offset: float = 0.02,
    show: bool = True,
) -> tuple[tuple[plt.Figure, plt.Axes], tuple[plt.Figure, plt.Axes]]:
    """
    Plot support and F1 score per topic, colouring bars by accuracy.
    """
  

    topics = label_rows.index.to_list()
    accuracy_values = label_rows[accuracy_col].clip(0, 1)
    vmin, vmax = float(accuracy_values.min()), float(accuracy_values.max())
    if np.isclose(vmin, vmax):
        vmin, vmax = 0.0, 1.0
    cmap = plt.cm.get_cmap(accuracy_cmap)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    accuracy_colors = cmap(norm(accuracy_values))

    if figsize is None:
        width, height = latex_set_size()
        figsize = (width * 2, height)

    rc = locale_parameters()
    sns.set(style="whitegrid", rc=rc)

    support_fig, support_ax = plt.subplots(figsize=figsize)
    support_bars = support_ax.bar(
        topics,
        label_rows[support_col],
        color=accuracy_colors,
    )
    support_ax.set_xlabel(xlabel,fontsize=14)
    support_ax.set_ylabel(support_ylabel,fontsize=14)
    support_ax.set_title(support_title,fontsize=18)
    max_support = float(label_rows[support_col].max())
    support_ax.set_ylim(0, max(1.0, max_support * 1.1))
    for bar, accuracy in zip(support_bars, accuracy_values):
        support_ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + accuracy_text_offset,
            f"{accuracy:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    support_sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    support_sm.set_array([])
    support_fig.colorbar(support_sm, ax=support_ax, label=accuracy_label)
    support_ax.tick_params(axis="x", rotation=xtick_rotation)
    support_fig.tight_layout()

    f1_fig, f1_ax = plt.subplots(figsize=figsize)
    f1_bars = f1_ax.bar(
        topics,
        label_rows[f1_col],
        color=accuracy_colors,
    )
    f1_ax.set_xlabel(xlabel,fontsize=14)
    f1_ax.set_ylabel(f1_ylabel,fontsize=14)
    f1_ax.set_title(f1_title,fontsize=18)
    f1_ax.set_ylim(0, 1.05)
    for bar, score in zip(f1_bars, label_rows[f1_col]):
        f1_ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + f1_text_offset,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    f1_sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    f1_sm.set_array([])
    f1_fig.colorbar(f1_sm, ax=f1_ax, label=accuracy_label)
    f1_ax.tick_params(axis="x", rotation=xtick_rotation)
    f1_fig.tight_layout()

    if support_output_path is not None:
        support_output_path = Path(support_output_path)
        support_output_path.parent.mkdir(parents=True, exist_ok=True)
        support_fig.savefig(support_output_path, dpi=300)

    if f1_output_path is not None:
        f1_output_path = Path(f1_output_path)
        f1_output_path.parent.mkdir(parents=True, exist_ok=True)
        f1_fig.savefig(f1_output_path, dpi=300)

    if show:
        plt.show()

    return (support_fig, support_ax), (f1_fig, f1_ax)
