# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Validate graph_trainer AOT compile with the RL model.

Builds a Qwen3 debug model, applies TP + AOT compile via the RL compile
pipeline, and runs forward + backward passes.  Compares loss and state-dict
keys against a non-compiled baseline to verify numeric correctness and
weight-naming compatibility.

Supports two modes controlled by the ``--cudagraph`` flag:

  AOT only (default):
    torchrun --nproc_per_node=2 torchtitan/experiments/rl/validate_compile.py

  AOT + cudagraph pass:
    torchrun --nproc_per_node=2 torchtitan/experiments/rl/validate_compile.py --cudagraph

The cudagraph path requires 3 forward/backward iterations (warmup, record,
replay) and calls ``cudagraph_teardown()`` before process group destruction
to avoid NCCL communicator leaks.
"""

import argparse
import os

import torch
import torch.distributed as dist

from torchtitan.config import CommConfig, TORCH_DTYPE_MAP
from torchtitan.config.configs import ParallelismConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.rl.compile import RLCompileConfig
from torchtitan.experiments.rl.models.parallelize import parallelize_qwen3
from torchtitan.models.qwen3 import model_registry
from torchtitan.tools import utils


def _build_model(model_spec, dtype, parallel_dims, parallelism, compile_config):
    """Build, parallelize, materialize, and init-weight a model."""
    with torch.device("meta"):
        with utils.set_default_dtype(dtype):
            model = model_spec.model.build()

    model = parallelize_qwen3(
        model,
        parallel_dims=parallel_dims,
        parallelism=parallelism,
        compile_config=compile_config,
    )
    model.to_empty(device=utils.device_type)
    with torch.no_grad():
        model.init_weights(buffer_device=None)
    return model


def _forward_backward(model, tokens):
    """Run a single forward + backward, return (logits, loss)."""
    logits = model(tokens)
    loss = logits.sum()
    loss.backward()
    return logits, loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cudagraph",
        action="store_true",
        help="Enable cudagraph compiler pass (requires NVIDIA GPU)",
    )
    args = parser.parse_args()

    device_module, device_type = utils.device_module, utils.device_type
    device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
    device_module.set_device(device)

    world_size = dist_utils.init_distributed(CommConfig())
    tp_degree = world_size  # use all GPUs for TP

    model_spec = model_registry("debugmodel")
    dtype = TORCH_DTYPE_MAP["bfloat16"]

    parallel_dims = ParallelDims(
        dp_shard=1,
        dp_replicate=1,
        cp=1,
        tp=tp_degree,
        pp=1,
        ep=1,
        etp=1,
        world_size=world_size,
    )
    parallel_dims.build_mesh()

    parallelism = ParallelismConfig(tensor_parallel_degree=tp_degree)

    # --- Compile config ---
    passes = ["cudagraph"] if args.cudagraph else []
    compile_config = RLCompileConfig(
        enable=True, mode="aot", backend="aot_eager", passes=passes
    )
    mode_label = "AOT + cudagraph" if args.cudagraph else "AOT"

    # --- Baseline (TP only, no compile) ---
    model_base = _build_model(model_spec, dtype, parallel_dims, parallelism, None)
    model_base.train()

    # --- Compiled model ---
    model_compiled = _build_model(
        model_spec, dtype, parallel_dims, parallelism, compile_config
    )
    model_compiled.load_state_dict(model_base.state_dict())
    model_compiled.train()

    # --- Forward + backward ---
    # CUDAGraph needs 3 iterations: warmup → record → replay.
    # For the non-cudagraph path a single iteration suffices, but we
    # run the same count to keep the comparison fair.
    num_iters = 3 if args.cudagraph else 1
    rank = dist.get_rank()
    vocab_size = model_spec.model.vocab_size
    seq_len = 64

    for i in range(num_iters):
        torch.manual_seed(42 + rank + i)
        tokens = torch.randint(0, vocab_size, (1, seq_len), device=device)

        # Zero grads before each iteration
        for p in model_base.parameters():
            p.grad = None
        for p in model_compiled.parameters():
            p.grad = None

        logits_base, loss_base = _forward_backward(model_base, tokens)
        logits_compiled, loss_compiled = _forward_backward(model_compiled, tokens)

        phase = {0: "warmup", 1: "record", 2: "replay"}.get(i, str(i))
        loss_match = torch.allclose(loss_base, loss_compiled)
        print(
            f"[rank {rank}] iter {i} ({phase}): "
            f"baseline={loss_base.item():.6f}, "
            f"compiled={loss_compiled.item():.6f}, "
            f"match={loss_match}"
        )

    # --- Verification (on the last iteration) ---
    shape_match = logits_base.shape == logits_compiled.shape
    print(
        f"[rank {rank}] Output shape baseline: {logits_base.shape}, "
        f"compiled: {logits_compiled.shape}, "
        f"Match: {shape_match}"
    )

    sd_base = model_base.state_dict()
    sd_compiled = model_compiled.state_dict()
    keys_match = set(sd_base.keys()) == set(sd_compiled.keys())
    print(f"[rank {rank}] State dict keys match: {keys_match}")

    if not keys_match:
        only_base = set(sd_base.keys()) - set(sd_compiled.keys())
        only_compiled = set(sd_compiled.keys()) - set(sd_base.keys())
        if only_base:
            print(f"[rank {rank}]   Only in baseline: {only_base}")
        if only_compiled:
            print(f"[rank {rank}]   Only in compiled: {only_compiled}")

    num_params_with_grad = sum(
        1 for p in model_compiled.parameters() if p.grad is not None
    )
    total_params = sum(1 for _ in model_compiled.parameters())
    print(f"[rank {rank}] Params with grad: {num_params_with_grad}/{total_params}")

    all_ok = loss_match and shape_match and keys_match
    print(f"\n[rank {rank}] {mode_label}: {'PASS' if all_ok else 'FAIL'}")

    # Teardown cudagraph pools before destroying the process group to
    # avoid NCCL communicator leaks (see Note [explicit cudagraph teardown]).
    if args.cudagraph:
        from torchtitan.experiments.graph_trainer.cudagraph import cudagraph_teardown

        cudagraph_teardown()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
