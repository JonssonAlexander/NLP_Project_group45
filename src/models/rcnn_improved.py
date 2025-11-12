"""
Helper code for strategically improved RCNN
"""

import torch
import pandas as pd
from src.models.rcnn import RCNNTextClassifier


def unpack_batch_rcnn(batch, device):
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
            for n, t in tensor_attrs.items():
                if t.dim() == 2 and t.dtype in (torch.long, torch.int64, torch.int32):
                    tokens = t
                    break

        if labels is None and tokens is not None:
            B = tokens.size(0)
            for n, t in tensor_attrs.items():
                if (
                    t.dim() == 1
                    and t.dtype in (torch.long, torch.int64, torch.int32)
                    and t.size(0) == B
                ):
                    labels = t
                    break

        if lengths is None and tokens is not None:
            B = tokens.size(0)
            for n, t in tensor_attrs.items():
                if (
                    t.dim() == 1
                    and t.size(0) == B
                    and t.dtype in (torch.long, torch.int64, torch.int32)
                    and (labels is None or not torch.equal(t, labels))
                ):
                    lengths = t
                    break

    tokens = tokens.to(device)
    labels = labels.to(device)

    if lengths is None:
        lengths = (tokens != 0).sum(dim=1)
    lengths = lengths.to(device)

    return tokens, lengths, labels


def split_train_test(loaders):
    if isinstance(loaders, dict):
        train = loaders.get("train") or loaders.get("train_loader")
        test = loaders.get("test") or loaders.get("test_loader")
    else:
        def pick(*names):
            for n in names:
                if hasattr(loaders, n):
                    return getattr(loaders, n)
            return None

        train = pick("train", "train_loader", "train_dataloader")
        test = pick("test", "test_loader", "test_dataloader")

    def ensure_iterable(dl, name):
        if callable(dl):
            dl = dl()
        return dl

    return ensure_iterable(train, "train"), ensure_iterable(test, "test")


def coarse_topic_accuracy_from_eval(eval_result, label_to_index):
    index_to_label = {idx: lbl for lbl, idx in label_to_index.items()}
    counts = {}

    y_true = eval_result["y_true"]
    y_pred = eval_result["y_pred"]

    for t, p in zip(y_true, y_pred):
        true_fine = index_to_label[int(t)]
        pred_fine = index_to_label[int(p)]

        true_coarse = true_fine.split(":")[0]
        pred_coarse = pred_fine.split(":")[0]

        if true_coarse not in counts:
            counts[true_coarse] = {"correct": 0, "total": 0}

        counts[true_coarse]["total"] += 1
        if pred_coarse == true_coarse:
            counts[true_coarse]["correct"] += 1

    rows = []
    order = ["ABBR", "ENTY", "NUM", "HUM", "LOC", "DESC"]
    for c in order:
        if c in counts:
            s = counts[c]
            acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
            rows.append({"label": c, "accuracy": acc, "support": s["total"]})
    for c, s in counts.items():
        if c not in order:
            acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
            rows.append({"label": c, "accuracy": acc, "support": s["total"]})

    return pd.DataFrame(rows)