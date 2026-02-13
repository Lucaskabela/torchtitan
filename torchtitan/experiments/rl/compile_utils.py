# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
RL-adapted compiler_toolkit integration.

Provides AOT export + joint graph compilation for RL policy models,
adapted from torchtitan.experiments.compiler_toolkit.graph_utils.
Key differences from the base compiler_toolkit:
  - No DTensor assertion (RL models use plain tensors, not FSDP/TP)
  - No job_config parameter (RL uses hardcoded config)
  - Input padding to fixed max_seq_len for stable AOT-traced shapes
"""

import contextlib
import functools
import logging
from typing import Any, Callable, List, Optional

import torch
import torch.nn.functional as F
from torch._dynamo.functional_export import dynamo_graph_capture_for_export
from torch._functorch.aot_autograd import (
    aot_compile_joint_with_descriptors,
    aot_export_joint_with_descriptors,
)
from torch._guards import tracing

from torchtitan.experiments.compiler_toolkit.graph_utils import (
    make_compiler_with_passes,
    validate_pass_names,
)

logger = logging.getLogger(__name__)


def rl_joint_graph_builder(
    model: torch.nn.Module,
    model_args: tuple,
    model_kwargs: dict,
    fw_compiler: Optional[Callable] = None,
    bw_compiler: Optional[Callable] = None,
    joint_custom_passes: Optional[List[Callable]] = None,
    dump_folder: str | None = None,
):
    """
    Build a joint forward-backward graph for an RL model.

    Adapted from compiler_toolkit.graph_utils.joint_graph_builder with
    the DTensor assertion and job_config parameter removed.

    Args:
        model: The model to compile
        model_args: Tuple of model input arguments (plain tensors)
        model_kwargs: Dict of model input keyword arguments
        fw_compiler: Optional custom forward compiler function
        bw_compiler: Optional custom backward compiler function
        joint_custom_passes: List of custom passes to run on the joint graph
        dump_folder: Optional folder to dump the graph to
    """
    assert isinstance(model_args, tuple)
    if model_kwargs is None:
        model_kwargs = {}

    # Inline export_joint to capture the Dynamo graph for parameter ordering
    with (
        torch._dynamo.config.patch(fake_tensor_cache_enabled=False),
        torch.fx.traceback.preserve_node_meta(),
    ):
        gm = dynamo_graph_capture_for_export(model)(*model_args, **model_kwargs)
        tracing_context = gm.meta["tracing_context"]

    # Collect parameters and buffers from gm in the same order that
    # aot_export_joint_with_descriptors uses internally.  The key is
    # remove_duplicate=False so that tied weights (e.g. tok_embeddings.weight
    # and output.weight pointing to the same tensor) are listed separately,
    # matching the primals the compiled function expects.
    graph_params = [p for _, p in gm.named_parameters(remove_duplicate=False)]
    graph_buffers = [b for _, b in gm.named_buffers(remove_duplicate=False)]

    with tracing(tracing_context):
        with contextlib.ExitStack() as stack:
            joint_with_descriptors = aot_export_joint_with_descriptors(
                stack,
                gm,
                model_args,
                model_kwargs,
            )

    # run custom passes on joint-graph before partitioner
    if joint_custom_passes is not None:
        for joint_custom_pass in joint_custom_passes:
            joint_with_descriptors.graph_module = joint_custom_pass(
                joint_with_descriptors.graph_module
            )

    with tracing(tracing_context):
        fn = aot_compile_joint_with_descriptors(
            joint_with_descriptors, fw_compiler=fw_compiler, bw_compiler=bw_compiler
        )

    def wrapper_fn(args, kwargs):
        inputs = [
            *graph_params,
            *graph_buffers,
            *args,
        ]
        return fn(*inputs, **kwargs)

    return wrapper_fn


class RLCompiledModule(torch.nn.Module):
    """
    Compiled module wrapper for RL policy models.

    Adapted from compiler_toolkit.graph_utils.CompiledModule with
    parallel_dims and parallelize_inputs removed (RL models don't use
    DTensor/FSDP/TP, only DDP in multi-process).

    Handles input padding to a fixed max_seq_len so that the AOT-traced
    graph always sees the same concrete shapes. Causal attention naturally
    handles right-padding.
    """

    def __init__(
        self,
        inner: torch.nn.Module,
        joint_graph_builder_fn: Callable,
        max_seq_len: int,
        cudagraph_warmup_iters: int = 0,
    ) -> None:
        super().__init__()
        self.inner = inner  # register as submodule
        self.joint_graph_builder_fn = joint_graph_builder_fn
        self.joint_graph_module = None
        self.max_seq_len = max_seq_len
        self.cudagraph_warmup_iters = cudagraph_warmup_iters

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)

    def state_dict(self, *args, **kwargs) -> Any:
        return self.inner.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs) -> Any:
        return self.inner.load_state_dict(*args, **kwargs)

    def named_parameters(self, *args, **kwargs) -> Any:
        return self.inner.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs) -> Any:
        return self.inner.parameters(*args, **kwargs)

    def forward(self, tokens, attention_masks=None):
        orig_len = tokens.shape[1]
        if orig_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length {orig_len} exceeds compile_max_seq_len "
                f"{self.max_seq_len}. Increase compile_max_seq_len or shorten inputs."
            )
        pad_len = self.max_seq_len - orig_len

        if pad_len > 0:
            padded_tokens = F.pad(tokens, (0, pad_len), value=0)
        else:
            padded_tokens = tokens

        if self.joint_graph_module is None:
            self.joint_graph_module = self.joint_graph_builder_fn(
                self.inner, (padded_tokens,), {}
            )
            # Run warmup iterations for CUDAGraph (warmup + record phases)
            # so that the first real training call uses replay mode.
            # The CUDAGraphWrapper needs 2 calls before replay: warmup (eager
            # on the graph capture stream) and record (captures the graph).
            # Without this, the warmup call runs on a different CUDA stream
            # which can produce divergent results.
            if self.cudagraph_warmup_iters > 0:
                logger.info(
                    f"Running {self.cudagraph_warmup_iters} CUDAGraph warmup iterations"
                )
                for _ in range(self.cudagraph_warmup_iters):
                    warmup_out = self.joint_graph_module((padded_tokens,), {})
                    warmup_out.sum().backward()
                # Clear gradients accumulated during warmup
                for p in self.inner.parameters():
                    if p.grad is not None:
                        p.grad.zero_()

        output = self.joint_graph_module((padded_tokens,), {})

        # Slice output back to original sequence length
        output = output[:, :orig_len, :]
        return output


def _ensure_custom_ops_registered():
    """Ensure vLLM compat custom ops are registered with torch.library.

    Importing these modules triggers @torch.library.custom_op registration
    for rms_norm, silu_and_mul, and flash_attn. These custom ops are opaque
    to both Dynamo and AOT autograd, so the compiled graph preserves them
    as single nodes that call vLLM kernels at runtime.
    """
    import torchtitan.experiments.rl.vllm_compat.batch_invariant_backward  # noqa: F401
    import torchtitan.experiments.rl.vllm_compat.models.attention  # noqa: F401


def compile_rl_model(
    model: torch.nn.Module,
    max_seq_len: int,
    joint_passes: Optional[List[str]] = None,
    compiler_passes: Optional[List[str]] = None,
    use_cudagraph: bool = False,
    dump_folder: Optional[str] = None,
) -> RLCompiledModule:
    """
    Compile an RL policy model using compiler_toolkit infrastructure.

    Args:
        model: The model to compile
        max_seq_len: Fixed sequence length for AOT tracing (inputs are padded to this)
        joint_passes: Optional list of joint pass names from AVAILABLE_JOINT_PASSES
        compiler_passes: Optional list of compiler pass names from AVAILABLE_COMPILER_PASSES
        use_cudagraph: If True, append the cudagraph pass (must be the last compiler pass)
        dump_folder: Optional folder to dump intermediate graphs

    Returns:
        RLCompiledModule wrapping the compiled model
    """
    _ensure_custom_ops_registered()

    if compiler_passes is None:
        compiler_passes = []
    if joint_passes is None:
        joint_passes = []

    if use_cudagraph and "cudagraph" not in compiler_passes:
        compiler_passes = list(compiler_passes) + ["cudagraph"]

    # Validate pass ordering and dependencies
    validate_pass_names(compiler_passes, joint_passes)

    # Resolve pass names to functions
    resolved_joint_passes = None
    if joint_passes:
        from torchtitan.experiments.compiler_toolkit.passes import (
            AVAILABLE_JOINT_PASSES,
        )

        resolved_joint_passes = []
        for name in joint_passes:
            if name not in AVAILABLE_JOINT_PASSES:
                raise ValueError(
                    f"Unknown joint pass: {name}. "
                    f"Available: {list(AVAILABLE_JOINT_PASSES.keys())}"
                )
            resolved_joint_passes.append(AVAILABLE_JOINT_PASSES[name])

    resolved_compiler_passes = None
    if compiler_passes:
        from torchtitan.experiments.compiler_toolkit.passes import (
            AVAILABLE_COMPILER_PASSES,
        )

        resolved_compiler_passes = []
        for name in compiler_passes:
            if name not in AVAILABLE_COMPILER_PASSES:
                raise ValueError(
                    f"Unknown compiler pass: {name}. "
                    f"Available: {list(AVAILABLE_COMPILER_PASSES.keys())}"
                )
            resolved_compiler_passes.append(AVAILABLE_COMPILER_PASSES[name])

    # Create fw/bw compilers with passes
    fw_compiler, bw_compiler = make_compiler_with_passes(
        passes=resolved_compiler_passes,
        dump_folder=dump_folder,
    )

    # Create the joint graph builder partial
    builder_fn = functools.partial(
        rl_joint_graph_builder,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        joint_custom_passes=resolved_joint_passes,
        dump_folder=dump_folder,
    )

    # CUDAGraph needs 2 warmup iterations (warmup + record) before replay
    cudagraph_warmup_iters = 2 if use_cudagraph else 0

    return RLCompiledModule(model, builder_fn, max_seq_len, cudagraph_warmup_iters)
