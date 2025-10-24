"""Utilities for token masking strategies used in text diffusion."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import math

import torch
from torch import Tensor


def _validate_ratio(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        msg = f"{name} must be in the range [0, 1], got {value}"
        raise ValueError(msg)


def sample_mask(
    input_shape: Tuple[int, ...],
    mask_ratio: float,
    *,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Sample a Bernoulli mask with the given ratio."""

    _validate_ratio("mask_ratio", mask_ratio)
    if mask_ratio == 0.0:
        return torch.zeros(input_shape, dtype=torch.bool, device=device)
    if mask_ratio == 1.0:
        return torch.ones(input_shape, dtype=torch.bool, device=device)

    rand = torch.rand(input_shape, generator=generator, device=device)
    return rand < mask_ratio


def apply_mask(input_ids: Tensor, mask: Tensor, mask_token_id: int) -> Tensor:
    """Replace masked positions with the mask token id."""

    if mask.dtype != torch.bool:
        mask = mask.bool()
    masked = input_ids.clone()
    masked[mask] = mask_token_id
    return masked


def mask_tokens(
    input_ids: Tensor,
    mask_ratio: float,
    mask_token_id: int,
    *,
    padding_mask: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor]:
    """Mask a fraction of tokens uniformly at random.

    Returns the masked token ids and the boolean mask that was applied.
    """

    mask = sample_mask(
        input_ids.shape,
        mask_ratio,
        device=input_ids.device,
        generator=generator,
    )
    if padding_mask is not None:
        mask &= padding_mask.bool()
    masked_tokens = apply_mask(input_ids, mask, mask_token_id)
    return masked_tokens, mask


def mask_ratio_from_timesteps(
    timesteps: Tensor,
    num_timesteps: int,
    *,
    schedule: str = "linear",
    custom_schedule: Optional[Callable[[Tensor, int], Tensor]] = None,
) -> Tensor:
    """Map timesteps to masking ratios using a simple schedule."""

    if num_timesteps <= 0:
        msg = "num_timesteps must be positive"
        raise ValueError(msg)
    if timesteps.ndim == 0:
        timesteps = timesteps.unsqueeze(0)
    if timesteps.ndim != 1:
        msg = "timesteps must be a 1D tensor"
        raise ValueError(msg)

    if custom_schedule is not None:
        ratios = custom_schedule(timesteps, num_timesteps)
    else:
        scaled = timesteps.to(dtype=torch.float32)
        if schedule == "linear":
            denom = max(num_timesteps - 1, 1)
            ratios = scaled / denom
        elif schedule == "cosine":
            frac = scaled / num_timesteps
            ratios = 1.0 - torch.cos(frac.clamp(0.0, 1.0) * math.pi / 2)
        else:
            msg = f"Unknown schedule '{schedule}'"
            raise ValueError(msg)

    return ratios.clamp(0.0, 1.0)


def mask_tokens_from_timesteps(
    input_ids: Tensor,
    timesteps: Tensor,
    num_timesteps: int,
    mask_token_id: int,
    *,
    schedule: str = "linear",
    custom_schedule: Optional[Callable[[Tensor, int], Tensor]] = None,
    padding_mask: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Mask tokens according to timestep-derived ratios."""

    if input_ids.ndim != 2:
        msg = "input_ids must be 2D (batch, seq_len)"
        raise ValueError(msg)
    if timesteps.shape[0] != input_ids.shape[0]:
        msg = "timesteps batch dimension must match input_ids"
        raise ValueError(msg)

    ratios = mask_ratio_from_timesteps(
        timesteps,
        num_timesteps,
        schedule=schedule,
        custom_schedule=custom_schedule,
    )

    rand = torch.rand(
        input_ids.shape,
        device=input_ids.device,
        dtype=torch.float32,
        generator=generator,
    )
    mask = rand < ratios.unsqueeze(-1)
    if padding_mask is not None:
        mask &= padding_mask.bool()
    masked_tokens = apply_mask(input_ids, mask, mask_token_id)
    return masked_tokens, mask, ratios


def sample_timesteps(
    num_timesteps: int,
    batch_size: int,
    *,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
    weights: Optional[Tensor] = None,
) -> Tensor:
    """Sample diffusion timesteps for a batch.

    When ``weights`` is provided, timesteps are drawn from the categorical
    distribution defined by ``weights``. Otherwise, sampling is uniform over
    ``[0, num_steps - 1]``.
    """

    if num_timesteps <= 0:
        msg = "num_steps must be positive"
        raise ValueError(msg)
    if batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)

    if weights is not None:
        if weights.ndim != 1 or weights.numel() != num_timesteps:
            msg = "weights must be a 1D tensor with length equal to num_steps"
            raise ValueError(msg)
        if torch.any(weights < 0):
            msg = "weights must be non-negative"
            raise ValueError(msg)
        probs = weights.to(dtype=torch.float32, device=device or weights.device)
        total = probs.sum()
        if total <= 0:
            msg = "weights must sum to a positive value"
            raise ValueError(msg)
        probs = probs / total
        cat = torch.distributions.Categorical(probs=probs)
        samples = cat.sample((batch_size,), generator=generator)
        if device is not None:
            samples = samples.to(device)
        return samples

    return torch.randint(
        0,
        num_timesteps,
        size=(batch_size,),
        device=device,
        generator=generator,
    )


def uniform_random_remask(
    tokens: Tensor,
    current_mask: Tensor,
    remask_ratio: float,
    mask_token_id: int,
    *,
    maskable: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor]:
    """Re-mask a random subset of the currently revealed tokens."""

    _validate_ratio("remask_ratio", remask_ratio)
    if tokens.shape != current_mask.shape:
        msg = "tokens and current_mask must share the same shape"
        raise ValueError(msg)

    if remask_ratio == 0.0:
        return tokens, current_mask

    if maskable is None:
        maskable = torch.ones_like(current_mask, dtype=torch.bool)
    else:
        maskable = maskable.bool()

    available = maskable & ~current_mask
    if not torch.any(available):
        return tokens, current_mask

    rand = torch.rand(tokens.shape, dtype=torch.float32, generator=generator, device=tokens.device)
    new_mask = (rand < remask_ratio) & available
    updated_mask = current_mask | new_mask
    updated_tokens = tokens.clone()
    updated_tokens[new_mask] = mask_token_id
    return updated_tokens, updated_mask


def confidence_based_remask(
    logits: Tensor,
    tokens: Tensor,
    current_mask: Tensor,
    mask_token_id: int,
    *,
    confidence_threshold: float,
    min_mask_tokens: int = 0,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Use model confidence to decide which masked positions stay masked."""

    _validate_ratio("confidence_threshold", confidence_threshold)
    if min_mask_tokens < 0:
        msg = "min_mask_tokens must be non-negative"
        raise ValueError(msg)
    if logits.shape[:2] != tokens.shape:
        msg = "logits and tokens must align on batch and sequence dimensions"
        raise ValueError(msg)

    probs = logits.softmax(dim=-1)
    max_probs, preds = probs.max(dim=-1)

    updated_tokens = tokens.clone()
    updated_mask = current_mask.clone().bool()

    masked_positions = updated_mask
    confident = masked_positions & (max_probs >= confidence_threshold)
    reconsider = masked_positions & ~confident

    updated_tokens[confident] = preds[confident]
    updated_mask[confident] = False
    updated_tokens[reconsider] = mask_token_id
    updated_mask[reconsider] = True

    if min_mask_tokens > 0:
        num_masked = updated_mask.sum(dim=-1)
        needs_more = num_masked < min_mask_tokens
        if torch.any(needs_more):
            batch_indices = torch.where(needs_more)[0]
            for batch_idx in batch_indices.tolist():
                candidate_positions = torch.where(~updated_mask[batch_idx])[0]
                if candidate_positions.numel() == 0:
                    continue
                required = int(min_mask_tokens - num_masked[batch_idx].item())
                if required <= 0:
                    continue
                perm = torch.randperm(
                    candidate_positions.numel(),
                    device=logits.device,
                    generator=generator,
                )
                chosen = candidate_positions[perm[:required]]
                updated_mask[batch_idx, chosen] = True
                updated_tokens[batch_idx, chosen] = mask_token_id

    return updated_tokens, updated_mask, max_probs
