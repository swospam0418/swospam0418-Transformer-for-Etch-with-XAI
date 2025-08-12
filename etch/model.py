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
    """Transformer model for Etch profiles with FiLM conditioning and masking."""

    def __init__(
        self,
        num_step_types: int,
        param_dim: int,
        profile_dim: int,
        d_model: int = 128,
        n_head: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_seq_len: int = 50,
    ) -> None:
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model

        # Base embeddings
        self.step_embedding = nn.Embedding(num_step_types + 1, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len + 1, d_model)

        # FiLM parameters for each step type
        self.film_gamma = nn.Embedding(num_step_types + 1, d_model)
        self.film_beta = nn.Embedding(num_step_types + 1, d_model)

        # Parameter projection without bias to allow masking
        self.param_projection = nn.Linear(param_dim, d_model, bias=False)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.input_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Expanded output head
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, d_model * 2),
            nn.ReLU(),
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, profile_dim),
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
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        step_seq: torch.Tensor,
        param_seq: torch.Tensor,
        mask: torch.Tensor,
        pos_seq: torch.Tensor | None = None,
        param_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            step_seq: ``(B, L)`` tensor of step type indices.
            param_seq: ``(B, L, P)`` tensor of step parameters. ``P`` is the
                maximum number of parameters across all step types.
            mask: ``(B, L)`` boolean tensor where ``True`` marks valid steps.
            pos_seq: Optional ``(B, L)`` tensor of position indices.
            param_mask: Optional ``(B, L, P)`` boolean tensor masking unused
                parameter entries.
        """

        B, L, _ = param_seq.shape

        # Step embeddings and FiLM parameters
        step_emb = self.step_embedding(step_seq)
        gamma = self.film_gamma(step_seq)
        beta = self.film_beta(step_seq)

        # Parameter projection with masking and FiLM modulation
        if param_mask is not None:
            param_seq = param_seq * param_mask
        param_emb = self.param_projection(param_seq)
        if param_mask is not None:
            denom = param_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(param_emb.dtype)
            param_emb = param_emb / denom
        param_emb = param_emb * gamma + beta

        x = step_emb + param_emb
        if pos_seq is not None:
            pos_emb = self.pos_embedding(pos_seq)
            x = x + pos_emb

        x = x / math.sqrt(self.d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Build attention mask
        cls_mask = torch.ones((B, 1), dtype=torch.bool, device=mask.device)
        key_padding_mask = ~(torch.cat([cls_mask, mask], dim=1))

        x = self.input_norm(x)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        cls_output = x[:, 0]
        return self.output_projection(cls_output)

