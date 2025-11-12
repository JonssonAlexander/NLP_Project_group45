"""
ENTY improved RCNN with contrastive auxiliary loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple
from src.models.rcnn import RCNNTextClassifier
from src.training import build_dataloaders_and_vocab
from src.evaluation import evaluate_model

PAD_IDX = 0  # adjust if your padding index differs

def unpack_batch_rcnn(batch, device, pad_idx: int = PAD_IDX):
    tokens = None
    lengths = None
    labels = None

    if isinstance(batch, dict):
        t = batch.get("text", batch.get("tokens", None))
        if isinstance(t, (tuple, list)) and len(t) == 2:
            tokens, lengths = t
        else:
            tokens = t
        for k in ("label", "labels", "y", "target", "targets"):
            if k in batch:
                labels = batch[k]
                break

    elif isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            tokens, labels = batch
        elif len(batch) >= 3:
            tokens, lengths, labels = batch[0], batch[1], batch[2]

    else:
        for name in ("text", "tokens", "inputs", "input_ids", "x"):
            if hasattr(batch, name):
                tokens = getattr(batch, name)
                break
        for name in ("label", "labels", "y", "target", "targets"):
            if hasattr(batch, name):
                labels = getattr(batch, name)
                break
        for name in ("lengths", "lens", "seq_lengths"):
            if hasattr(batch, name):
                lengths = getattr(batch, name)
                break

        tensor_attrs = {
            n: getattr(batch, n)
            for n in dir(batch)
            if not n.startswith("_")
            and hasattr(batch, n)
            and torch.is_tensor(getattr(batch, n))
        }

        if tokens is None:
            for _, t in tensor_attrs.items():
                if t.dim() == 2 and t.dtype in (torch.long, torch.int64, torch.int32):
                    tokens = t
                    break

        if labels is None and tokens is not None:
            B = tokens.size(0)
            for _, t in tensor_attrs.items():
                if (
                    t.dim() == 1
                    and t.dtype in (torch.long, torch.int64, torch.int32)
                    and t.size(0) == B
                ):
                    labels = t
                    break

        if lengths is None and tokens is not None:
            B = tokens.size(0)
            for _, t in tensor_attrs.items():
                if (
                    t.dim() == 1
                    and t.size(0) == B
                    and t.dtype in (torch.long, torch.int64, torch.int32)
                    and (labels is None or not torch.equal(t, labels))
                ):
                    lengths = t
                    break

    if tokens is None or labels is None:
        raise ValueError(f"Could not infer tokens/labels from batch of type {type(batch)}.")

    tokens = tokens.to(device)
    labels = labels.to(device)

    if lengths is None:
        lengths = (tokens != pad_idx).sum(dim=1)
    lengths = lengths.to(device)

    return tokens, lengths, labels

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

def build_dataloaders(config, splits, vocab_report):
    loaders, label_to_index = build_dataloaders_and_vocab(
        config=config,
        splits=splits,
        vocabulary=vocab_report.vocabulary,
    )
    return loaders, label_to_index


class CoarseContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        B = features.size(0)
        if B < 2:
            return torch.tensor(0.0, device=device)

        dist = torch.cdist(features, features, p=2)
        same = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        pos_loss = (dist * same).pow(2).sum() / (same.sum() + 1e-6)
        neg_loss = ((F.relu(self.margin - dist)) * (1 - same)).pow(2).sum() / ((1 - same).sum() + 1e-6)
        return pos_loss + neg_loss


def train_enty(loaders, config, device, embedding_tensor, label_to_index, alpha: float = 1.0, beta: float = 0.5):

    num_classes = len(label_to_index)

    model = RCNNTextClassifier(
        embedding_matrix=embedding_tensor.clone(),
        num_classes=num_classes,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        padding_idx=PAD_IDX,
        multisample_dropout=4,
    ).to(device)

    ce_loss = nn.CrossEntropyLoss()
    contrastive_loss = CoarseContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=getattr(config, "learning_rate", 1e-3))

    model.train()
    history = {"train_loss": []}

    for epoch in range(getattr(config, "num_epochs", 10)):
        total_loss = 0.0
        n_batches = 0

        for batch in loaders.train:
            optimizer.zero_grad()

            # robust batch unpacking
            tokens, lengths, labels = unpack_batch_rcnn(batch, device, pad_idx=PAD_IDX)

            logits = model(tokens, lengths)

            # use internal features if available; otherwise fallback to logits
            features = getattr(model, "rnn_output", None)
            if features is None:
                features = logits.detach()

            loss_cls = ce_loss(logits, labels)
            loss_con = contrastive_loss(features, labels)
            loss = alpha * loss_cls + beta * loss_con

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        history["train_loss"].append(total_loss / max(1, n_batches))
        print(f"Epoch {epoch+1} | Loss: {history['train_loss'][-1]:.4f}")

    eval_result = evaluate_model(model, loaders.test, device=device)
    return history, model, eval_result

