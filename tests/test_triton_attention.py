import math

import pytest
import torch

from src.triton_attention import triton_attention


@pytest.mark.parametrize("seq_len", [16, 37])
@pytest.mark.parametrize("heads", [2, 4])
@pytest.mark.parametrize("d_head", [32, 64])
@pytest.mark.parametrize("padded_fraction", [0.0, 0.25, 0.5])
def test_triton_attention_matches_mha(seq_len, heads, d_head, padded_fraction):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton attention test")

    torch.manual_seed(0)
    batch = 2
    device = torch.device("cuda")

    qkv = torch.randn(batch, heads, seq_len, d_head, device=device, dtype=torch.float16)
    key_padding_mask = None
    if padded_fraction > 0:
        mask = torch.zeros(batch, seq_len, dtype=torch.bool, device=device)
        pad_tokens = int(seq_len * padded_fraction)
        if pad_tokens > 0:
            mask[:, -pad_tokens:] = True
        key_padding_mask = mask

    mha = torch.nn.MultiheadAttention(embed_dim=heads * d_head, num_heads=heads, batch_first=True).to(device)
    mha = mha.to(dtype=torch.float32)

    qkv_flat = qkv.permute(0, 2, 1, 3).reshape(batch, seq_len, heads * d_head).to(torch.float32)
    mha_eval, _ = mha(qkv_flat, qkv_flat, qkv_flat, key_padding_mask=key_padding_mask)
    mha_eval = mha_eval.view(batch, seq_len, heads, d_head).permute(0, 2, 1, 3)

    triton_eval = triton_attention(qkv, qkv, qkv, key_padding_mask)
    triton_eval = triton_eval.to(torch.float32)

    tol = 1e-2
    assert torch.allclose(triton_eval, mha_eval, atol=tol, rtol=tol)
