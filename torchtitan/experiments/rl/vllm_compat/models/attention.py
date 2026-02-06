# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
VLLM-compatible Flash Attention wrapper for torch.compile.

The vLLM FA3 C extension (``_vllm_fa3_C.fwd``) has no Meta/fake-tensor
implementation, so ``torch.compile`` cannot trace through it.  We use
``torch.library.custom_op`` to register an opaque operator whose forward
calls the real kernel and whose backward is a pure-PyTorch re-computation
of causal attention.  This makes the whole operator invisible to both the
Dynamo frontend *and* the AOTAutograd backend while still participating in
the compiled graph.
"""

import math

import torch

# ---------------------------------------------------------------------------
# Resolve vLLM helpers once at import time
# ---------------------------------------------------------------------------
from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_varlen_func,
    get_flash_attn_version,
)

_FA_VERSION = get_flash_attn_version()


# ---------------------------------------------------------------------------
# Custom op: forward calls vLLM flash-attn, backward is pure PyTorch
# ---------------------------------------------------------------------------


@torch.library.custom_op("vllm_compat::flash_attn_varlen_fwd", mutates_args=())
def flash_attn_varlen_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    softmax_scale: float,
    num_splits: int,
) -> torch.Tensor:
    """Forward-only flash-attention via vLLM's kernel."""
    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=softmax_scale,
        causal=True,
        num_splits=num_splits,
        fa_version=_FA_VERSION,
    )


@flash_attn_varlen_fwd.register_fake
def _flash_attn_varlen_fwd_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    softmax_scale: float,
    num_splits: int,
) -> torch.Tensor:
    """Fake (meta-tensor) implementation — just return the right shape/dtype."""
    return torch.empty_like(q)


def _flash_attn_backward(ctx, grad_output):
    """Pure-PyTorch causal attention backward (no vLLM C extensions needed)."""
    q, k, v = ctx.saved_tensors
    scale = ctx.softmax_scale
    seq_len = ctx.max_seqlen

    total_tokens, num_heads, head_dim = q.shape
    batch_size = total_tokens // seq_len

    q_batch = q.reshape(batch_size, seq_len, num_heads, head_dim)
    k_batch = k.reshape(batch_size, seq_len, k.shape[1], head_dim)
    v_batch = v.reshape(batch_size, seq_len, v.shape[1], head_dim)
    grad_out_batch = grad_output.reshape(batch_size, seq_len, num_heads, head_dim)

    # Transpose to (batch, heads, seq, dim)
    q_t = q_batch.transpose(1, 2)
    k_t = k_batch.transpose(1, 2)
    v_t = v_batch.transpose(1, 2)
    grad_out_t = grad_out_batch.transpose(1, 2)

    # GQA: expand K/V heads to match Q's num_heads
    num_kv_heads = k.shape[1]
    if num_kv_heads < num_heads:
        n_rep = num_heads // num_kv_heads
        k_t = k_t.repeat_interleave(n_rep, dim=1)
        v_t = v_t.repeat_interleave(n_rep, dim=1)

    # QK^T * scale
    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale

    # Causal mask
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
        diagonal=1,
    )
    scores = scores.masked_fill(causal_mask, float("-inf"))

    attn_weights = torch.nn.functional.softmax(scores, dim=-1)

    # grad_v = attn_weights^T @ grad_out
    grad_v_t = torch.matmul(attn_weights.transpose(-2, -1), grad_out_t)

    # grad_attn_weights = grad_out @ v^T
    grad_attn_weights = torch.matmul(grad_out_t, v_t.transpose(-2, -1))

    # Backward through softmax
    sum_term = (grad_attn_weights * attn_weights).sum(dim=-1, keepdim=True)
    grad_scores = attn_weights * (grad_attn_weights - sum_term)
    grad_scores = grad_scores.masked_fill(causal_mask, 0.0)
    grad_scores = grad_scores * scale

    grad_q_t = torch.matmul(grad_scores, k_t)
    grad_k_t = torch.matmul(grad_scores.transpose(-2, -1), q_t)

    # GQA: reduce gradients back to num_kv_heads
    if num_kv_heads < num_heads:
        n_rep = num_heads // num_kv_heads
        grad_k_t = grad_k_t.reshape(
            batch_size, num_kv_heads, n_rep, seq_len, head_dim
        ).sum(dim=2)
        grad_v_t = grad_v_t.reshape(
            batch_size, num_kv_heads, n_rep, seq_len, head_dim
        ).sum(dim=2)

    grad_q = grad_q_t.transpose(1, 2).reshape(total_tokens, num_heads, head_dim)
    grad_k = grad_k_t.transpose(1, 2).reshape(total_tokens, k.shape[1], head_dim)
    grad_v = grad_v_t.transpose(1, 2).reshape(total_tokens, v.shape[1], head_dim)

    # Return grads for (q, k, v, cu_seqlens, max_seqlen, softmax_scale, num_splits)
    return grad_q, grad_k, grad_v, None, None, None, None


def _flash_attn_setup_context(ctx, inputs, output):
    q, k, v, cu_seqlens, max_seqlen, softmax_scale, num_splits = inputs
    ctx.save_for_backward(q, k, v)
    ctx.max_seqlen = max_seqlen
    ctx.softmax_scale = softmax_scale


torch.library.register_autograd(
    "vllm_compat::flash_attn_varlen_fwd",
    _flash_attn_backward,
    setup_context=_flash_attn_setup_context,
)


# ---------------------------------------------------------------------------
# nn.Module wrapper used by the rest of the model
# ---------------------------------------------------------------------------


class VLLMCompatibleFlashAttention(torch.nn.Module):
    """Wrapper around FlashAttention as used by VLLM."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        # Input is (batch, num_heads, seq_len, head_dim) — transpose to
        # (batch, seq_len, num_heads, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        batch_size, seq_len, num_heads, head_dim = q.shape
        num_kv_heads = k.shape[2]

        # Flatten to varlen format: (total_tokens, num_heads, head_dim)
        q_varlen = q.reshape(-1, num_heads, head_dim)
        k_varlen = k.reshape(-1, num_kv_heads, head_dim)
        v_varlen = v.reshape(-1, num_kv_heads, head_dim)

        cu_seqlens = torch.arange(
            0,
            (batch_size + 1) * seq_len,
            seq_len,
            dtype=torch.int32,
            device=q.device,
        )

        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        num_splits = 1 if vllm_is_batch_invariant() else 0

        output_varlen = torch.ops.vllm_compat.flash_attn_varlen_fwd(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens,
            seq_len,
            scale,
            num_splits,
        )

        # Reshape back: (total_tokens, num_heads, head_dim) →
        # (batch, seq_len, num_heads, head_dim)
        output = output_varlen.reshape(batch_size, seq_len, num_heads, head_dim)

        # Transpose back to (batch, num_heads, seq_len, head_dim)
        output = output.transpose(1, 2)
        return output
