# %% [markdown]
# # 02 - RNN Baseline Training
#
# Notebook scope:
# - reuse the prepared embeddings and tokenised data
# - train a simple RNN classifier with pooled sentence representations
# - explore regularisation strategies and sentence representation variants
# - report validation curves, best config, and topic-wise accuracy

# %% [markdown]
# ## Imports & data setup

# %%
from dataclasses import replace
from pathlib import Path
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm, colors
from IPython.display import Markdown, display

from src.config import load_data_config
from src.dataset_pipeline import prepare_tokenised_splits
from src.embeddings import load_torchtext_glove
from src.reports import build_vocabulary_report
from src.training import RNNExperimentConfig, train_rnn_model
from src.evaluation import (
    evaluate_model,
    topic_accuracy_table,
    classification_report_table,
)
from src.plotting import plot_training_curves
import torch

SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
sns.set(style="whitegrid")

data_config = load_data_config(Path("configs/data.yaml"))
splits = prepare_tokenised_splits(data_config)
len(splits.train), len(splits.validation), len(splits.test)

# %% [markdown]
# ## Embedding prep

# %%
vocab_report = build_vocabulary_report(
    tokenised_dataset=splits.train,
    min_freq=data_config.vocabulary_min_freq,
    specials=data_config.vocabulary_specials,
)

embedding_result = load_torchtext_glove(
    vocabulary=vocab_report.vocabulary,
    name="6B",
    dim=100,
    trainable=True,
    random_seed=7,
)

# %% [markdown]
# ## Baseline configuration

# %%
baseline_config = RNNExperimentConfig(
    epochs=15,
    batch_size=64,
    hidden_dim=128,
    num_layers=1,
    dropout=0.3,
    learning_rate=1e-3,
    weight_decay=0.0,
    grad_clip=1.0,
    pooling="last_hidden",
    optimizer="adam",
    early_stopping_patience=3,
)

# %% [markdown]
# ## Train baseline RNN

# %%
history, model, label_to_index, dataloaders = train_rnn_model(
    config=baseline_config,
    splits=splits,
    vocabulary=vocab_report.vocabulary,
    embedding_result=embedding_result,
)
index_to_label = {idx: label for label, idx in label_to_index.items()}
history.best_epoch, len(history.train_loss)

# %% [markdown]
# ## Training curves

# %%
plot_training_curves(
    history,
    title="Baseline RNN training dynamics",
    output_path=Path("plots/part2_rnn_baseline_curves.png"),
);

# %%
peak_val_epoch = history.best_epoch
peak_val_accuracy = max(history.val_accuracy)
display(Markdown(f"Validation accuracy peaks at epoch {peak_val_epoch} with {peak_val_accuracy:.3f}, after which the curve plateaus and training loss keeps falling slowly."))

# %% [markdown]
# ## Validation & test metrics

# %%
val_eval = evaluate_model(model, dataloaders.validation)
test_eval = evaluate_model(model, dataloaders.test)

baseline_metrics = pd.DataFrame(
    [
        {"split": "validation", "loss": val_eval.loss, "accuracy": val_eval.accuracy},
        {"split": "test", "loss": test_eval.loss, "accuracy": test_eval.accuracy},
    ]
)
baseline_metrics

# %% [markdown]
# ## Topic-wise accuracy

# %%
topic_accuracy_df = topic_accuracy_table(test_eval, index_to_label)

fig, ax = plt.subplots(figsize=(8, 5))
accuracy_colors = [cm.get_cmap("viridis")(acc) for acc in topic_accuracy_df["accuracy"].clip(0, 1)]
bars = ax.bar(topic_accuracy_df["label"], topic_accuracy_df["support"], color=accuracy_colors)
ax.set_xlabel("Topic")
ax.set_ylabel("Support")
ax.set_title("Topic-wise support (color = accuracy)")
ax.set_ylim(0, topic_accuracy_df["support"].max() * 1.1)
for bar, accuracy in zip(bars, topic_accuracy_df["accuracy"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 5,
        f"{accuracy:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
sm = plt.cm.ScalarMappable(cmap=cm.get_cmap("viridis"), norm=colors.Normalize(0, 1))
sm.set_array([])
fig.colorbar(sm, ax=ax, label="Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
fig.savefig("plots/part2_topic_accuracy.png", dpi=300)
plt.show()

# %%
lowest_topics = topic_accuracy_df.sort_values('accuracy').head(2)
notes = ', '.join(f"{row.label} ({row.accuracy:.2f})" for row in lowest_topics.itertuples())
display(Markdown(f"Topic-wise accuracies flag the weakest labels: {notes}."))

# %% [markdown]
# ## Classification report

# %%
class_report_df = classification_report_table(test_eval, index_to_label)
label_rows = class_report_df.loc[
    ~class_report_df.index.isin(["accuracy", "macro avg", "weighted avg", "micro avg"])
].copy()

if "accuracy" not in label_rows.columns:
    label_rows["accuracy"] = label_rows.get("recall", 0.0)

fig, ax = plt.subplots(figsize=(8, 5))
support_colors = [cm.get_cmap("magma")(acc) for acc in label_rows["accuracy"].clip(0, 1)]
bars = ax.bar(label_rows.index, label_rows["support"], color=support_colors)
ax.set_xlabel("Topic")
ax.set_ylabel("Support")
ax.set_title("Support by topic (color = accuracy)")
ax.set_ylim(0, label_rows["support"].max() * 1.1)
for bar, accuracy in zip(bars, label_rows["accuracy"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 5,
        f"{accuracy:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
sm = plt.cm.ScalarMappable(cmap=cm.get_cmap("magma"), norm=colors.Normalize(0, 1))
sm.set_array([])
fig.colorbar(sm, ax=ax, label="Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
fig.savefig("plots/part2_classification_support.png", dpi=300)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
f1_colors = [cm.get_cmap("plasma")(score) for score in label_rows["f1-score"].clip(0, 1)]
bars = ax.bar(label_rows.index, label_rows["f1-score"], color=f1_colors)
ax.set_xlabel("Topic")
ax.set_ylabel("F1 score")
ax.set_title("F1 score by topic")
ax.set_ylim(0, 1.05)
for bar, score in zip(bars, label_rows["f1-score"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{score:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
plt.xticks(rotation=45)
plt.tight_layout()
fig.savefig("plots/part2_classification_f1.png", dpi=300)
plt.show()

averages_df = class_report_df.loc[
    ["accuracy", "macro avg", "weighted avg"],
    [col for col in class_report_df.columns if col in ["precision", "recall", "f1-score", "support"]],
]
averages_df

# %% [markdown]
# ## Regularisation comparison

# %%
regularisation_variants = [
    ("baseline", baseline_config),
    ("no_regularisation", replace(baseline_config, dropout=0.0, weight_decay=0.0, grad_clip=0.0)),
    ("dropout_0.0", replace(baseline_config, dropout=0.0)),
    ("dropout_0.5", replace(baseline_config, dropout=0.5)),
    ("weight_decay_1e-4", replace(baseline_config, weight_decay=1e-4)),
    ("grad_clip_off", replace(baseline_config, grad_clip=0.0)),
]

reg_results = []
for name, cfg in regularisation_variants:
    hist, mdl, _, loaders = train_rnn_model(
        config=cfg,
        splits=splits,
        vocabulary=vocab_report.vocabulary,
        embedding_result=embedding_result,
    )
    val_res = evaluate_model(mdl, loaders.validation)
    test_res = evaluate_model(mdl, loaders.test)
    reg_results.append(
        {
            "name": name,
            "dropout": cfg.dropout,
            "weight_decay": cfg.weight_decay,
            "grad_clip": cfg.grad_clip,
            "best_epoch": hist.best_epoch,
            "val_accuracy": val_res.accuracy,
            "test_accuracy": test_res.accuracy,
        }
    )

regularisation_df = pd.DataFrame(reg_results).sort_values(
    "test_accuracy", ascending=False
)
regularisation_df

# %%
best_regulariser = regularisation_df.iloc[0]
control = regularisation_df[regularisation_df['name'] == 'no_regularisation'].iloc[0]
display(Markdown(f"Best regularisation setup: {best_regulariser['name']} (test accuracy {best_regulariser['test_accuracy']:.3f}) versus no regularisation at {control['test_accuracy']:.3f}."))

# %% [markdown]
# ## Sentence representation strategies

# %%
pooling_options = ["last_hidden", "mean", "max", "attention"]
pooling_results = []

for pooling in pooling_options:
    cfg = replace(baseline_config, pooling=pooling)
    hist, mdl, _, loaders = train_rnn_model(
        config=cfg,
        splits=splits,
        vocabulary=vocab_report.vocabulary,
        embedding_result=embedding_result,
    )
    test_res = evaluate_model(mdl, loaders.test)
    pooling_results.append(
        {
            "pooling": pooling,
            "best_epoch": hist.best_epoch,
            "test_accuracy": test_res.accuracy,
        }
    )

pooling_df = pd.DataFrame(pooling_results).sort_values(
    "test_accuracy", ascending=False
)
pooling_df

# %%
best_pooling = pooling_df.iloc[0]
display(Markdown(f"Top pooling strategy: {best_pooling['pooling']} with test accuracy {best_pooling['test_accuracy']:.3f}. Differences across methods stay within {pooling_df['test_accuracy'].max() - pooling_df['test_accuracy'].min():.3f} absolute points."))

# %% [markdown]
# ## Wrap-up

# %%
summary = pd.DataFrame(
    [
        {
            "hidden_dim": baseline_config.hidden_dim,
            "layers": baseline_config.num_layers,
            "dropout": baseline_config.dropout,
            "learning_rate": baseline_config.learning_rate,
            "grad_clip": baseline_config.grad_clip,
            "pooling": baseline_config.pooling,
            "best_epoch": history.best_epoch,
            "val_accuracy": val_eval.accuracy,
            "test_accuracy": test_eval.accuracy,
        }
    ]
)
summary

# %%
config_row = summary.iloc[0]
display(Markdown(
    f"Best configuration => epochs {baseline_config.epochs}, hidden_dim {config_row['hidden_dim']}, batch size {baseline_config.batch_size}, learning rate {baseline_config.learning_rate}, optimizer {baseline_config.optimizer}, dropout {config_row['dropout']}, grad clip {baseline_config.grad_clip}.\n"
    f"Validation accuracy {config_row['val_accuracy']:.3f}; test accuracy {config_row['test_accuracy']:.3f}."
))

# %%
