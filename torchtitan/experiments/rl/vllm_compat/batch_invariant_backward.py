# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Batch-invariant operations with backward pass support.

This module adds gradient support to vLLM's deterministic batch_invariant mode
by registering backward operations that also use vLLM's deterministic kernels.

Key architecture:
- Forward: Uses vLLM's batch_invariant Triton kernels (deterministic)
- Backward: Also uses vLLM's batch_invariant kernels (deterministic)

This achieves bitwise-deterministic RL training where both rollouts (forward)
and training (forward + backward) produce identical results.

The ``SiluAndMul`` and ``RMSNorm`` forwards call vLLM C/Triton kernels that
lack Meta/fake-tensor implementations.  They are wrapped with
``torch.library.custom_op`` so that ``torch.compile(fullgraph=True)`` can
handle them without tracing into unsupported operators.

Usage:
    from vllm.model_executor.layers.batch_invariant import init_batch_invariance
    from batch_invariant_backward import enable_batch_invariant_backward_mode

    # Initialize vLLM's deterministic mode first
    init_batch_invariance(AttentionBackendEnum.FLASH_ATTN)

    # Then enable gradient support
    enable_batch_invariant_backward_mode()

    # Now all operations are deterministic AND support gradients
    model = MyModel()
    output = model(input)  # deterministic forward
    loss = compute_loss(output)
    loss.backward()  # gradients work with deterministic backward!
"""

import torch

# ============================================================================
# Eager vLLM helpers — resolved once at import time
# ============================================================================

from vllm.model_executor.layers.batch_invariant import rms_norm as _vllm_rms_norm


def _vllm_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Call vLLM's CUDA ``silu_and_mul`` kernel."""
    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(out, x)
    return out


# ============================================================================
# SiluAndMul custom op
# ============================================================================


@torch.library.custom_op("vllm_compat::silu_and_mul_fwd", mutates_args=())
def silu_and_mul_fwd(x: torch.Tensor) -> torch.Tensor:
    """Forward: vLLM's CUDA SiluAndMul kernel."""
    return _vllm_silu_and_mul(x)


@silu_and_mul_fwd.register_fake
def _silu_and_mul_fwd_fake(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    return torch.empty(output_shape, dtype=x.dtype, device=x.device)


def _silu_and_mul_backward(ctx, grad_output):
    (x,) = ctx.saved_tensors
    d = x.shape[-1] // 2
    gate = x[..., :d]
    up = x[..., d:]

    sigmoid_gate = torch.sigmoid(gate)
    silu_gate = gate * sigmoid_gate

    # d_silu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    d_silu_gate = sigmoid_gate * (1 + gate * (1 - sigmoid_gate))

    grad_gate = grad_output * up * d_silu_gate
    grad_up = grad_output * silu_gate

    grad_x = torch.cat([grad_gate, grad_up], dim=-1)
    return (grad_x,)


def _silu_and_mul_setup_context(ctx, inputs, output):
    (x,) = inputs
    ctx.save_for_backward(x)


torch.library.register_autograd(
    "vllm_compat::silu_and_mul_fwd",
    _silu_and_mul_backward,
    setup_context=_silu_and_mul_setup_context,
)


# ============================================================================
# RMSNorm custom op
# ============================================================================


@torch.library.custom_op("vllm_compat::rms_norm_fwd", mutates_args=())
def rms_norm_fwd(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Forward: vLLM's Triton RMSNorm kernel."""
    return _vllm_rms_norm(input, weight, eps)


@rms_norm_fwd.register_fake
def _rms_norm_fwd_fake(
    input: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    return torch.empty_like(input)


def _rms_norm_backward(ctx, grad_output):
    input, weight = ctx.saved_tensors
    eps = ctx.eps

    variance = (input * input).mean(dim=-1, keepdim=True)
    rms = torch.sqrt(variance + eps)
    x_norm = input / rms

    grad_weight = (grad_output * x_norm).sum(dim=tuple(range(grad_output.ndim - 1)))

    grad_x_norm = grad_output * weight
    mean_term = (grad_x_norm * x_norm).mean(dim=-1, keepdim=True)
    grad_input = (grad_x_norm - mean_term * x_norm) / rms

    return grad_input, grad_weight, None


def _rms_norm_setup_context(ctx, inputs, output):
    input, weight, eps = inputs
    ctx.save_for_backward(input, weight)
    ctx.eps = eps


torch.library.register_autograd(
    "vllm_compat::rms_norm_fwd",
    _rms_norm_backward,
    setup_context=_rms_norm_setup_context,
)


# ============================================================================
# Backward operation implementations for autograd
# ============================================================================


def matmul_backward_impl(grad_output, self, other, output_mask):
    """
    Backward pass for matmul: y = matmul(a, b)
    Returns: (grad_a, grad_b)

    Args:
        grad_output: Gradient from downstream
        self: First input tensor (a)
        other: Second input tensor (b)
        output_mask: List of bools indicating which gradients to compute [self, other]

    grad_a = grad_output @ b.T
    grad_b = a.T @ grad_output

    Uses torch.matmul which is overridden by vLLM's batch_invariant mode!
    """
    grad_self = grad_other = None

    compute_grad_self = output_mask[0] if len(output_mask) > 0 else True
    compute_grad_other = output_mask[1] if len(output_mask) > 1 else True

    if compute_grad_self:
        if other.ndim == 2:
            grad_self = torch.matmul(grad_output, other.t())
        elif other.ndim == 3:
            grad_self = torch.matmul(grad_output, other.transpose(-2, -1))
        else:
            grad_self = torch.matmul(grad_output, other.transpose(-2, -1))

    if compute_grad_other:
        if self.ndim == 2:
            grad_other = torch.matmul(self.t(), grad_output)
        elif self.ndim == 3:
            grad_other = torch.matmul(self.transpose(-2, -1), grad_output)
        else:
            grad_other = torch.matmul(self.transpose(-2, -1), grad_output)

    return grad_self, grad_other


def linear_backward_impl(grad_output, input, weight, output_mask):
    """
    Backward pass for linear: y = input @ weight.T + bias
    Returns: (grad_input, grad_weight, grad_bias)

    PyTorch passes args in weird order: (saved_input, grad_output, weight, output_mask)
    So we swap the first two args in our implementation.
    """
    input, grad_output = grad_output, input

    grad_input = grad_weight = grad_bias = None

    compute_grad_input = output_mask[0] if len(output_mask) > 0 else True
    compute_grad_weight = output_mask[1] if len(output_mask) > 1 else True
    compute_grad_bias = output_mask[2] if len(output_mask) > 2 else True

    if compute_grad_input:
        grad_input = torch.matmul(grad_output, weight)

    if compute_grad_weight:
        if input.ndim == 3:
            input_2d = input.reshape(-1, input.shape[-1])
            grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
            grad_weight = torch.matmul(grad_output_2d.transpose(0, 1), input_2d)
        else:
            grad_weight = torch.matmul(grad_output.transpose(0, 1), input)

    if compute_grad_bias:
        grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))

    return grad_input, grad_weight, grad_bias


# ============================================================================
# Registration
# ============================================================================

_batch_invariant_backward_mode = False
_batch_invariant_backward_lib = None


def enable_batch_invariant_backward_mode():
    """Enable batch invariant backward mode to support gradients.

    This function adds backward pass support to vLLM's existing batch_invariant
    implementations by registering the backward operations. vLLM handles all the
    forward passes, we just add gradient support.
    """
    global _batch_invariant_backward_mode, _batch_invariant_backward_lib

    if _batch_invariant_backward_mode:
        return

    from vllm.model_executor.layers import batch_invariant as vllm_bi

    if (
        not hasattr(vllm_bi, "_batch_invariant_LIB")
        or vllm_bi._batch_invariant_LIB is None
    ):
        raise RuntimeError(
            "vLLM's batch_invariant mode is not initialized. "
            "Call init_batch_invariance(AttentionBackendEnum.FLASH_ATTN) first."
        )

    _batch_invariant_backward_lib = vllm_bi._batch_invariant_LIB

    _batch_invariant_backward_lib.impl(
        "aten::matmul_backward", matmul_backward_impl, "CUDA"
    )
    _batch_invariant_backward_lib.impl(
        "aten::linear_backward", linear_backward_impl, "CUDA"
    )

    _batch_invariant_backward_mode = True


def disable_batch_invariant_backward_mode():
    """Disable batch invariant backward mode."""
    global _batch_invariant_backward_mode, _batch_invariant_backward_lib

    if _batch_invariant_backward_lib is not None:
        _batch_invariant_backward_lib._destroy()

    _batch_invariant_backward_mode = False
    _batch_invariant_backward_lib = None


def is_batch_invariant_backward_mode_enabled():
    """Check if batch invariant backward mode is enabled."""
    return _batch_invariant_backward_mode


# ============================================================================
# Public API for gradient-enabled vLLM operations
# ============================================================================


def rms_norm_with_gradients(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    RMS normalization with gradient support.

    Uses vLLM's Triton kernel for forward pass (deterministic) and
    batch-invariant PyTorch operations for backward pass.
    """
    return torch.ops.vllm_compat.rms_norm_fwd(input, weight, eps)


def silu_and_mul_with_gradients(x: torch.Tensor) -> torch.Tensor:
    """
    SiluAndMul activation with gradient support.

    Uses vLLM's implementation for forward pass (deterministic) and
    implements proper backward pass for training.
    """
    return torch.ops.vllm_compat.silu_and_mul_fwd(x)
