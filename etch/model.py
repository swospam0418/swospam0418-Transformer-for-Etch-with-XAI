from __future__ import annotations
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class EtchTransformer(nn.Module):
    """Simplified Transformer model for Etch profiles."""

    def __init__(self, num_step_types: int, param_dim: int, profile_dim: int,
                 d_model: int = 128, n_head: int = 4, num_layers: int = 3,
                 dropout: float = 0.1, max_seq_len: int = 50) -> None:
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.step_embedding = nn.Embedding(num_step_types + 1, d_model, padding_idx=0)
        self.param_projection = nn.Sequential(
            nn.Linear(param_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
        self.input_norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head,
                                                   dim_feedforward=d_model * 4,
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, profile_dim)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, step_seq: torch.Tensor, param_seq: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L = step_seq.shape
        step_emb = self.step_embedding(step_seq)
        param_emb = self.param_projection(param_seq)
        x = (step_emb + param_emb) / math.sqrt(self.d_model)
        cls_tokens = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        if mask is not None:
            cls_mask = torch.ones((B, 1), dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)
            key_padding_mask = ~mask
        else:
            key_padding_mask = None
        x = self.input_norm(x)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        cls_output = x[:, 0]
        return self.output_projection(cls_output)

