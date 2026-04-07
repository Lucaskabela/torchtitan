# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from typing import TypeAlias

import torch
import torch.nn as nn
import torch.utils.checkpoint

from torchtitan.config import CompileConfig
from torchtitan.tools.logging import logger

# PyTorch's default ignore index for cross-entropy loss
IGNORE_INDEX = -100

LossFunction: TypeAlias = Callable[..., torch.Tensor]


def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss with sum reduction for token-based normalization."""
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(),
        labels.flatten(0, 1),
        reduction="sum",
        ignore_index=IGNORE_INDEX,
    )


def build_cross_entropy_loss(compile_config: CompileConfig, **kwargs):
    del kwargs  # delete any unused arguments
    loss_fn = cross_entropy_loss
    if compile_config.enable and "loss" in compile_config.components:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn, backend=compile_config.backend)
    return loss_fn


class OutputHeadAndLoss(nn.Module):
    """Fuses norm + output projection + cross-entropy loss into a single module.

    With inductor, the auto_chunker FX pass automatically chunks the vocabulary
    projection along the batch dimension. With other backends, manual chunking
    with gradient checkpointing avoids materializing the full logits tensor.

    The norm and output_proj modules are NOT registered as submodules -- they
    remain owned by the model for FSDP/checkpoint purposes.
    """

    _norm: nn.Module
    _output_proj: nn.Module
    _num_chunks: int

    def __init__(
        self, norm: nn.Module, output_proj: nn.Module, num_chunks: int = 0
    ) -> None:
        super().__init__()
        # Bypass nn.Module.__setattr__ to avoid registering as submodules
        object.__setattr__(self, "_norm", norm)
        object.__setattr__(self, "_output_proj", output_proj)
        object.__setattr__(self, "_num_chunks", num_chunks)

    def _compute_chunk_loss(
        self, hidden_chunk: torch.Tensor, label_chunk: torch.Tensor
    ) -> torch.Tensor:
        logits = self._output_proj(hidden_chunk)
        return torch.nn.functional.cross_entropy(
            logits.flatten(0, 1).float(),
            label_chunk.flatten(0, 1),
            reduction="sum",
            ignore_index=IGNORE_INDEX,
        )

    def forward(
        self, hidden_states: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        h = self._norm(hidden_states)

        if self._num_chunks <= 1:
            return self._compute_chunk_loss(h, labels)

        # Manual chunking: process chunks sequentially with gradient
        # checkpointing to avoid materializing the full logits tensor.
        h_chunks = h.flatten(0, 1).chunk(self._num_chunks)
        label_chunks = labels.flatten(0, 1).chunk(self._num_chunks)
        loss = torch.tensor(0.0, device=h.device, dtype=torch.float32)
        for h_chunk, label_chunk in zip(h_chunks, label_chunks):
            # pyrefly: ignore [bad-assignment]
            chunk_loss: torch.Tensor = torch.utils.checkpoint.checkpoint(
                self._compute_chunk_loss,
                h_chunk.unsqueeze(0),
                label_chunk.unsqueeze(0),
                use_reentrant=False,
            )
            loss = loss + chunk_loss
        return loss


_DEFAULT_MANUAL_NUM_CHUNKS = 4


def build_chunked_loss(
    model: nn.Module, compile_config: CompileConfig
) -> OutputHeadAndLoss:
    """Build a compiled OutputHeadAndLoss.

    With inductor backend, uses auto_chunker. With other backends, uses
    manual chunking with gradient checkpointing.
    """
    use_auto_chunker = compile_config.backend == "inductor"
    num_chunks = (
        0
        if use_auto_chunker
        else (compile_config.chunked_loss_num_chunks or _DEFAULT_MANUAL_NUM_CHUNKS)
    )
    # pyrefly: ignore [bad-argument-type]
    output_head_and_loss = OutputHeadAndLoss(model.norm, model.output, num_chunks)

    options: dict = {}
    if use_auto_chunker:
        options["auto_chunker.enable"] = True
        if compile_config.chunked_loss_num_chunks is not None:
            options["auto_chunker.num_chunk"] = compile_config.chunked_loss_num_chunks
        logger.info(
            "Compiling OutputHeadAndLoss with inductor auto_chunker "
            f"(num_chunks={compile_config.chunked_loss_num_chunks})"
        )
    else:
        logger.info(
            f"Compiling OutputHeadAndLoss with manual chunking "
            f"(num_chunks={num_chunks})"
        )

    # pyrefly: ignore [bad-return]
    return torch.compile(
        output_head_and_loss,
        backend=compile_config.backend,
        fullgraph=True,
        options=options if options else None,
    )


def mse_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Common MSE loss function with sum reduction for Transformer models training."""
    return torch.nn.functional.mse_loss(
        pred.float(), labels.float().detach(), reduction="sum"
    )


def build_mse_loss(compile_config: CompileConfig, **kwargs):
    del kwargs  # delete any unused arguments
    loss_fn = mse_loss
    if compile_config.enable and "loss" in compile_config.components:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn, backend=compile_config.backend)
    return loss_fn
