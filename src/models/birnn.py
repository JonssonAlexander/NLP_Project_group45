"""
Bidirectional recurrent models used in Part 3 enhancements.
"""

from __future__ import annotations

import torch
from torch import nn


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM classifier that concatenates the final forward and backward
    hidden states before classification.
    """

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=False, padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=embedding_matrix.size(1),
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
        representation = self.dropout(representation)
        return self.fc(representation)


class BiGRUClassifier(nn.Module):
    """
    Bidirectional GRU classifier mirroring the LSTM variant.
    """

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=False, padding_idx=0
        )
        self.gru = nn.GRU(
            input_size=embedding_matrix.size(1),
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
        representation = self.dropout(representation)
        return self.fc(representation)
