"""
Recurrent CNN for Part 3.3
"""

from __future__ import annotations
import torch
import torch.nn as nn

class RCNNTextClassifier(nn.Module):
    """
    RCNN: BiGRU encodes contextual token representations -> concatenate with raw embeddings
    -> apply linear 1D convolution + tanh -> masked global max-pool -> classification. 
    """
    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        padding_idx: int = 0,
        multisample_dropout: int = 0,
    ) -> None:
        super().__init__()
        vocab_size, emb_dim = embedding_matrix.size()
        self.padding_idx = padding_idx
        self.multisample_dropout = multisample_dropout
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=False, padding_idx=padding_idx
        )
        self.bigru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.feature_dim = hidden_dim * 2 + emb_dim
        self.linear = nn.Linear(self.feature_dim, hidden_dim * 2)
        self.embed_dropout2d = nn.Dropout2d(p=dropout)  
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens) 
        embedded_sd = self.embed_dropout2d(embedded.transpose(1, 2)).transpose(1, 2)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded_sd, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.bigru(packed)
        H, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True) 

        feats = torch.cat([H, embedded_sd], dim=-1)
        feats = torch.tanh(self.linear(feats)) 

        B, T, D = feats.size()
        device = feats.device
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        mask = positions < lengths.unsqueeze(1)                  
        feats_masked = feats.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        pooled = torch.amax(feats_masked, dim=1)

        if self.training and self.multisample_dropout and self.multisample_dropout > 0:
            logits_sum = 0
            for _ in range(self.multisample_dropout):
                logits_sum += self.fc(self.dropout(pooled))
            return logits_sum / float(self.multisample_dropout)
        return self.fc(self.dropout(pooled))
