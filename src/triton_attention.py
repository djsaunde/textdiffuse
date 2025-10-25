"""Triton kernels for masked multi-head attention."""

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    mask_ptr,
    out_ptr,
    stride_q_bh, stride_q_m, stride_q_d,
    stride_k_bh, stride_k_m, stride_k_d,
    stride_v_bh, stride_v_m, stride_v_d,
    stride_mask_b, stride_mask_m,
    stride_out_bh, stride_out_m, stride_out_d,
    n_heads,
    n_ctx: tl.constexpr,
    d_head: tl.constexpr,
    has_mask: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    bh_id = tl.program_id(0)
    b_id = tl.idiv(bh_id, n_heads)
    start_m = tl.program_id(1) * BLOCK_M

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, d_head)

    q_ptrs = q_ptr + bh_id * stride_q_bh + offs_m[:, None] * stride_q_m + offs_d[None, :] * stride_q_d
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < n_ctx), other=0.0)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, d_head), dtype=tl.float32)

    for start_n in range(0, n_ctx, BLOCK_N):
        k_ptrs = k_ptr + bh_id * stride_k_bh + (start_n + offs_n)[None, :] * stride_k_m + offs_d[:, None] * stride_k_d
        v_ptrs = v_ptr + bh_id * stride_v_bh + (start_n + offs_n)[:, None] * stride_v_m + offs_d[None, :] * stride_v_d
        k = tl.load(k_ptrs, mask=(start_n + offs_n)[None, :] < n_ctx, other=0.0)
        v = tl.load(v_ptrs, mask=(start_n + offs_n)[:, None] < n_ctx, other=0.0)

        attn_scores = tl.dot(q, k, allow_tf32=True) / math.sqrt(d_head)

        if has_mask:
            mask_ptrs = mask_ptr + b_id * stride_mask_b + (start_n + offs_n)[None, :] * stride_mask_m
            mask_vals = tl.load(mask_ptrs, mask=(start_n + offs_n)[None, :] < n_ctx, other=0.0)
            attn_scores += mask_vals

        m_i_new = tl.maximum(m_i, tl.max(attn_scores, axis=1))
        p = tl.exp(attn_scores - m_i_new[:, None])
        l_i = l_i * tl.exp(m_i - m_i_new) + tl.sum(p, axis=1)
        acc = acc * (tl.exp(m_i - m_i_new)[:, None]) + tl.dot(p, v)
        m_i = m_i_new

    acc = acc / l_i[:, None]
    out_ptrs = out_ptr + bh_id * stride_out_bh + offs_m[:, None] * stride_out_m + offs_d[None, :] * stride_out_d
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < n_ctx)


def triton_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor] = None,
    *,
    block_m: int = 32,
    block_n: int = 64,
) -> torch.Tensor:
    """Compute attention with a per-token mask using a Triton kernel.

    Args:
        q: tensor of shape (batch, heads, seq_len, head_dim)
        k: tensor of shape (batch, heads, seq_len, head_dim)
        v: tensor of shape (batch, heads, seq_len, head_dim)
        key_padding_mask: optional boolean mask (batch, seq_len)
    """

    assert q.is_cuda and k.is_cuda and v.is_cuda, "Triton attention requires CUDA tensors"
    batch, heads, seq_len, head_dim = q.shape
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    output = torch.empty_like(q)

    if key_padding_mask is not None:
        mask = key_padding_mask.to(device=q.device, dtype=torch.bool)
        neg_inf = torch.tensor(-float("inf"), device=q.device, dtype=q.dtype)
        zero = torch.tensor(0.0, device=q.device, dtype=q.dtype)
        mask = torch.where(mask, neg_inf, zero)
    else:
        mask = torch.zeros((1, seq_len), device=q.device, dtype=q.dtype)

    q_bh = q.view(batch * heads, seq_len, head_dim)
    k_bh = k.view(batch * heads, seq_len, head_dim)
    v_bh = v.view(batch * heads, seq_len, head_dim)
    out_bh = output.view(batch * heads, seq_len, head_dim)

    grid = (batch * heads, triton.cdiv(seq_len, block_m))

    _attention_kernel[grid](
        q_bh,
        k_bh,
        v_bh,
        mask,
        out_bh,
        q_bh.stride(0), q_bh.stride(1), q_bh.stride(2),
        k_bh.stride(0), k_bh.stride(1), k_bh.stride(2),
        v_bh.stride(0), v_bh.stride(1), v_bh.stride(2),
        mask.stride(0) if key_padding_mask is not None else 0,
        mask.stride(1) if key_padding_mask is not None else 0,
        out_bh.stride(0), out_bh.stride(1), out_bh.stride(2),
        n_heads=heads,
        n_ctx=seq_len,
        d_head=head_dim,
        has_mask=key_padding_mask is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=2,
    )
    return output
