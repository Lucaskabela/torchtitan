# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Minimal reproduction of Triton 3.7 LLVM SLP vectorizer crash.

Repro (single GPU):
    rm -rf ~/.triton/cache /tmp/torchinductor_$USER
    python triton_slp_repro.py

Workaround: prepend DISABLE_LLVM_OPT=1
"""

import os

os.environ.pop("DISABLE_LLVM_OPT", None)

import torch
from torch.nn.attention.flex_attention import create_block_mask

_compiled_create_block_mask = torch.compile(create_block_mask)


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def main():
    print("Building attention mask...")
    block_mask = _compiled_create_block_mask(causal_mask, 8, None, 2048, 2048)
    print("Done — no crash.")


if __name__ == "__main__":
    main()
