"""
Convolutional sentence classifier inspired by Kim (2014).
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CNNTextClassifier(nn.Module):
    """
    Multi-filter CNN with max-over-time pooling.
    """

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        num_classes: int,
        filter_sizes: tuple[int, ...] = (3, 4, 5),
        num_filters: int = 100,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=False, padding_idx=0
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=num_filters,
                    kernel_size=(fs, embedding_matrix.size(1)),
                )
                for fs in filter_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(
        self, tokens: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        embedded = self.embedding(tokens).unsqueeze(1)  # (batch, 1, seq, dim)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(feature, feature.size(2)).squeeze(2) for feature in conved]
        representation = torch.cat(pooled, dim=1)
        representation = self.dropout(representation)
        return self.fc(representation)
