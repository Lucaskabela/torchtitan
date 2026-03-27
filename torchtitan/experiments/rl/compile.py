# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
RL-specific compile configuration and dispatch.

Extends graph_trainer's AOT compilation pipeline for use with the RL
PolicyTrainer. The key difference from core torchtitan's per-block
``torch.compile`` is that graph_trainer captures the entire model as a
single joint forward-backward graph, enabling whole-model compiler passes
(e.g. cudagraphs).
"""

from dataclasses import dataclass

import torch.nn as nn

from torchtitan.config import ParallelismConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig


@dataclass(kw_only=True, slots=True)
class RLCompileConfig(GraphTrainerCompileConfig):
    """Compile config for RL trainer, extending graph_trainer's config.

    Defaults to AOT mode with ``aot_eager`` backend for numeric stability
    during RL training. Set ``mode=None`` to fall through to the existing
    per-block compile path in ``parallelize_qwen3``.
    """

    mode: str | None = "aot"
    backend: str = "aot_eager"


def apply_compile_rl(
    model: nn.Module,
    *,
    compile_config: RLCompileConfig,
    parallelism: ParallelismConfig,
    parallel_dims: ParallelDims,
    dump_folder: str = "",
) -> nn.Module:
    """Apply graph_trainer's compile pipeline to the RL model.

    This is a thin wrapper around ``graph_trainer.compile.apply_compile``
    that provides the RL-appropriate call signature.

    Returns:
        The compiled model (``CompiledModule`` wrapper for AOT mode).
    """
    return apply_compile(
        model,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )
