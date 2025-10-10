"""
Recurrent neural network classifier for sentence-level topic prediction.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


PoolingType = Literal["last_hidden", "mean", "max", "attention"]


class RNNClassifier(nn.Module):
    """
    Baseline RNN classifier with configurable pooling strategy.
    """

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        pooling: PoolingType = "last_hidden",
        trainable_embeddings: bool = True,
    ) -> None:
        super().__init__()
        embedding_dim = embedding_matrix.size(1)

        self.embedding = nn.Embedding.from_pretrained(
            embeddings=embedding_matrix,
            freeze=not trainable_embeddings,
            padding_idx=0,
        )

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.pooling: PoolingType = pooling
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, num_classes)

        if pooling == "attention":
            self.attention = nn.Linear(hidden_dim, 1, bias=False)
        else:
            self.attention = None

    def forward(self, batch_tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        batch_tokens : torch.Tensor
            LongTensor of shape (batch, seq_len) containing token indices.
        lengths : torch.Tensor
            LongTensor of actual lengths before padding.
        """

        embedded = self.embedding(batch_tokens)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, hidden = self.rnn(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        if self.pooling == "last_hidden":
            representation = hidden[-1]
        elif self.pooling == "mean":
            representation = self._mean_pool(output, lengths)
        elif self.pooling == "max":
            representation = self._max_pool(output, lengths)
        elif self.pooling == "attention":
            representation = self._attention_pool(output, lengths)
        else:  # pragma: no cover - safeguarded by type hints
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        representation = self.dropout(representation)
        logits = self.output(representation)
        return logits

    def _mean_pool(self, outputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = self._mask_from_lengths(outputs, lengths)
        masked = outputs * mask.unsqueeze(-1)
        summed = masked.sum(dim=1)
        lengths = lengths.clamp(min=1).unsqueeze(1).to(outputs.device)
        return summed / lengths

    def _max_pool(self, outputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = self._mask_from_lengths(outputs, lengths)
        masked = outputs.masked_fill(~mask.unsqueeze(-1), -1e9)
        return masked.max(dim=1).values

    def _attention_pool(self, outputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if self.attention is None:
            raise RuntimeError("Attention layer not initialised.")

        mask = self._mask_from_lengths(outputs, lengths)
        scores = self.attention(outputs).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), outputs)
        return context.squeeze(1)

    @staticmethod
    def _mask_from_lengths(outputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        max_len = outputs.size(1)
        device = outputs.device
        range_tensor = torch.arange(max_len, device=device).unsqueeze(0)
        return range_tensor < lengths.unsqueeze(1)
