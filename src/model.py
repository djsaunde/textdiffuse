"""Skeleton implementation of a text diffusion model with bidirectional attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .triton_attention import triton_attention


@dataclass
class DiffusionModelConfig:
    """Configuration options for the text diffusion model."""

    vocab_size: int
    max_seq_len: int
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    ff_mult: float = 4.0
    dropout: float = 0.0
    timestep_embed_dim: Optional[int] = None


class SinusoidalTimestepEmbedding(nn.Module):
    """Generates sinusoidal embeddings for diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: Tensor) -> Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=device, dtype=torch.float32)
        exponent = -torch.log(torch.tensor(10000.0, device=device)) * exponent / max(half_dim - 1, 1)
        args = timesteps.float().unsqueeze(-1) * torch.exp(exponent)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class TransformerBlock(nn.Module):
    """Standard decoder-only transformer block with bidirectional attention."""

    def __init__(self, config: DiffusionModelConfig) -> None:
        super().__init__()
        self.num_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.use_triton = torch.cuda.is_available()
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.ff_norm = nn.LayerNorm(config.d_model)
        ff_hidden = int(config.d_model * config.ff_mult)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, ff_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(ff_hidden, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        residual = x
        attn_input = self.attn_norm(x)
        use_triton = self.use_triton and attn_input.is_cuda
        if use_triton:
            qkv = attn_input.view(attn_input.size(0), attn_input.size(1), self.num_heads, self.head_dim)
            qkv = qkv.permute(0, 2, 1, 3)
            attn_output = triton_attention(qkv, qkv, qkv, key_padding_mask)
            attn_output = attn_output.permute(0, 2, 1, 3).reshape_as(attn_input)
        else:
            qkv = attn_input.view(attn_input.size(0), attn_input.size(1), self.num_heads, self.head_dim)
            qkv = qkv.permute(0, 2, 1, 3)
            attn_output = F.scaled_dot_product_attention(
                qkv,
                qkv,
                qkv,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
                key_padding_mask=key_padding_mask,
            )
            attn_output = attn_output.permute(0, 2, 1, 3).reshape_as(attn_input)
        x = residual + attn_output
        residual = x
        x = self.ff_norm(x)
        x = residual + self.ff(x)
        return x


class TextDiffusionModel(nn.Module):
    """Minimal text diffusion backbone with configurable transformer layers."""

    def __init__(self, config: DiffusionModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embed = nn.Embedding(config.max_seq_len, config.d_model)

        timestep_dim = config.timestep_embed_dim or config.d_model
        self.timestep_embed = SinusoidalTimestepEmbedding(timestep_dim)
        self.timestep_proj = nn.Sequential(
            nn.Linear(timestep_dim, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)

    def forward(
        self,
        input_ids: Tensor,
        timesteps: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass returning the predicted noise residual."""

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()

        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            msg = "Sequence length exceeds configured maximum"
            raise ValueError(msg)

        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        token_embeddings = self.token_embed(input_ids)
        position_embeddings = self.position_embed(position_ids)
        hidden_states = token_embeddings + position_embeddings

        timestep_embeddings = self.timestep_embed(timesteps)
        timestep_embeddings = self.timestep_proj(timestep_embeddings)
        hidden_states = hidden_states + timestep_embeddings.unsqueeze(1)

        key_padding_mask = ~attention_mask
        for block in self.blocks:
            hidden_states = block(hidden_states, key_padding_mask=key_padding_mask)

        hidden_states = self.final_norm(hidden_states)
        noise_pred = self.output_proj(hidden_states)
        return noise_pred
