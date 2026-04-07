# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn
from torchtitan.components.loss import (
    cross_entropy_loss,
    IGNORE_INDEX,
    OutputHeadAndLoss,
)


class TestOutputHeadAndLoss(unittest.TestCase):
    def _make_modules(self, dim=32, vocab_size=64):
        norm = nn.LayerNorm(dim)
        output_proj = nn.Linear(dim, vocab_size, bias=False)
        return norm, output_proj

    def test_no_submodule_registration(self):
        """OutputHeadAndLoss must not register norm/output as submodules."""
        norm, output_proj = self._make_modules()
        module = OutputHeadAndLoss(norm, output_proj)

        self.assertEqual(list(module.named_children()), [])
        self.assertEqual(list(module.parameters()), [])

    def test_numerics(self):
        """OutputHeadAndLoss must match separate norm + output + cross_entropy_loss."""
        torch.manual_seed(42)
        dim, vocab_size = 32, 64
        B, T = 2, 8

        norm, output_proj = self._make_modules(dim, vocab_size)
        hidden = torch.randn(B, T, dim)
        labels = torch.randint(0, vocab_size, (B, T))
        labels[0, 1] = IGNORE_INDEX

        # Path A: separate
        h = norm(hidden)
        logits = output_proj(h)
        loss_a = cross_entropy_loss(logits, labels)

        # Path B: fused (no manual chunking)
        module = OutputHeadAndLoss(norm, output_proj)
        loss_b = module(hidden, labels)

        torch.testing.assert_close(loss_a, loss_b)

    def test_numerics_manual_chunking(self):
        """Manual chunking must match non-chunked numerics."""
        torch.manual_seed(42)
        dim, vocab_size = 32, 64
        B, T = 2, 8

        norm, output_proj = self._make_modules(dim, vocab_size)
        hidden = torch.randn(B, T, dim)
        labels = torch.randint(0, vocab_size, (B, T))
        labels[0, 1] = IGNORE_INDEX

        h = norm(hidden)
        logits = output_proj(h)
        loss_expected = cross_entropy_loss(logits, labels)

        for num_chunks in (2, 4):
            module = OutputHeadAndLoss(norm, output_proj, num_chunks=num_chunks)
            loss_chunked = module(hidden, labels)
            torch.testing.assert_close(loss_expected, loss_chunked)

    def test_gradients(self):
        """Gradients must match between fused and separate paths."""
        torch.manual_seed(42)
        dim, vocab_size = 32, 64
        B, T = 2, 8

        norm, output_proj = self._make_modules(dim, vocab_size)
        hidden = torch.randn(B, T, dim, requires_grad=True)
        labels = torch.randint(0, vocab_size, (B, T))

        # Path A: separate
        h = norm(hidden)
        logits = output_proj(h)
        loss_a = cross_entropy_loss(logits, labels)
        loss_a.backward()
        grad_hidden_a = hidden.grad.clone()
        grad_weight_a = output_proj.weight.grad.clone()
        grad_norm_a = norm.weight.grad.clone()

        # Reset
        hidden.grad = None
        output_proj.weight.grad = None
        norm.weight.grad = None

        # Path B: fused
        module = OutputHeadAndLoss(norm, output_proj)
        loss_b = module(hidden, labels)
        loss_b.backward()

        torch.testing.assert_close(grad_hidden_a, hidden.grad)
        torch.testing.assert_close(grad_weight_a, output_proj.weight.grad)
        torch.testing.assert_close(grad_norm_a, norm.weight.grad)

    def test_gradients_manual_chunking(self):
        """Gradients must match between manual chunked and separate paths."""
        torch.manual_seed(42)
        dim, vocab_size = 32, 64
        B, T = 2, 8

        norm, output_proj = self._make_modules(dim, vocab_size)
        hidden = torch.randn(B, T, dim, requires_grad=True)
        labels = torch.randint(0, vocab_size, (B, T))

        # Path A: separate
        h = norm(hidden)
        logits = output_proj(h)
        loss_a = cross_entropy_loss(logits, labels)
        loss_a.backward()
        grad_hidden_a = hidden.grad.clone()
        grad_weight_a = output_proj.weight.grad.clone()
        grad_norm_a = norm.weight.grad.clone()

        # Reset
        hidden.grad = None
        output_proj.weight.grad = None
        norm.weight.grad = None

        # Path B: manual chunking
        module = OutputHeadAndLoss(norm, output_proj, num_chunks=4)
        loss_b = module(hidden, labels)
        loss_b.backward()

        torch.testing.assert_close(grad_hidden_a, hidden.grad)
        torch.testing.assert_close(grad_weight_a, output_proj.weight.grad)
        torch.testing.assert_close(grad_norm_a, norm.weight.grad)


class TestDecoderReturnHiddenStates(unittest.TestCase):
    def test_flag_changes_output(self):
        """Setting _return_hidden_states should return pre-norm hidden states."""
        from torchtitan.models.llama3 import llama3_configs

        config = llama3_configs["debugmodel"]()
        with torch.device("meta"):
            model = config.build()

        # Move to CPU for testing
        model.to_empty(device="cpu")
        model.init_weights(buffer_device=torch.device("cpu"))
        model.eval()

        B, T = 1, 16
        tokens = torch.randint(0, config.vocab_size, (B, T))

        with torch.no_grad():
            # Default: returns logits [B, T, vocab_size]
            output = model(tokens)
            self.assertEqual(output.shape, (B, T, config.vocab_size))

            # With flag: returns hidden states [B, T, dim]
            model._return_hidden_states = True
            hidden = model(tokens)
            self.assertEqual(hidden.shape, (B, T, config.dim))


if __name__ == "__main__":
    unittest.main()
