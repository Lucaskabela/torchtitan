# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from torchtitan.models.common.attention import VarlenMetadata


# TODO We should either unify all the mask creation for RL, or move them to a
#      single file.
def build_varlen_metadata(
    input_sequences: list[tuple[torch.Tensor, int, int]], device: torch.device
) -> VarlenMetadata:
    """Build VarlenMetadata for all sequences in a batch."""
    cu_seqs = torch.cumsum(
        torch.tensor(
            [0] + [token_ids.shape[0] for token_ids, _, _ in input_sequences],
            dtype=torch.int32,
            device=device,
        ),
        0,
        dtype=torch.int32,
    )
    max_len = max(token_ids.shape[0] for token_ids, _, _ in input_sequences)
    return VarlenMetadata(
        cu_seq_q=cu_seqs, cu_seq_k=cu_seqs, max_q=max_len, max_k=max_len
    )


_DEFAULT_RL_NUM_CHUNKS = 4


def _chunk_output_and_gather(
    output_proj: torch.nn.Module,
    h_chunk: torch.Tensor,
    target_chunk: torch.Tensor,
) -> torch.Tensor:
    """Compute output projection, log_softmax, and gather for one chunk."""
    logits = output_proj(h_chunk)
    log_probs = F.log_softmax(logits.to(torch.float32), dim=-1)
    return log_probs.gather(2, target_chunk.unsqueeze(-1)).squeeze(-1)


def chunked_compute_token_log_probs(
    model: torch.nn.Module,
    prompt_ids: list[int],
    gen_ids: list[int],
    device: torch.device,
    num_chunks: int = _DEFAULT_RL_NUM_CHUNKS,
) -> torch.Tensor:
    """Memory-efficient compute_token_log_probs via chunked output projection.

    Requires ``model._return_hidden_states = True`` so the model returns
    hidden states before norm + output. Chunks the vocab projection to avoid
    materializing the full ``[seq_len, vocab_size]`` logits tensor.
    """
    token_ids = torch.tensor(prompt_ids + gen_ids, dtype=torch.long, device=device)
    prompt_len = len(prompt_ids)
    gen_len = len(gen_ids)
    attention_masks = build_varlen_metadata([(token_ids, prompt_len, gen_len)], device)

    full_tensor = token_ids.unsqueeze(0)
    seq_len = full_tensor.shape[1]
    positions = torch.arange(seq_len, device=device).unsqueeze(0)

    # Model returns hidden states (before norm + output projection)
    hidden = model(full_tensor, attention_masks=attention_masks, positions=positions)

    # Norm is cheap ([1, seq_len, dim]) -- apply on full tensor
    h = model.norm(hidden)

    # Chunk along sequence dim for the output projection
    h_shifted = h[:, :-1, :]  # [1, seq_len-1, dim]
    target_tokens = full_tensor[:, 1:]  # [1, seq_len-1]

    h_chunks = h_shifted.chunk(num_chunks, dim=1)
    target_chunks = target_tokens.chunk(num_chunks, dim=1)

    all_gathered = []
    for h_chunk, t_chunk in zip(h_chunks, target_chunks):
        gathered = torch.utils.checkpoint.checkpoint(
            _chunk_output_and_gather,
            model.output,
            h_chunk,
            t_chunk,
            use_reentrant=False,
        )
        all_gathered.append(gathered)

    all_log_probs = torch.cat(all_gathered, dim=1).squeeze(0)  # [seq_len-1]

    gen_start_idx = prompt_len - 1
    gen_end_idx = gen_start_idx + gen_len
    return all_log_probs[gen_start_idx:gen_end_idx]


def compute_token_log_probs(
    model: torch.nn.Module,
    prompt_ids: list[int],
    gen_ids: list[int],
    device: torch.device,
) -> torch.Tensor:
    """
    Compute per-token log probabilities for generated tokens.
    TODO Only batch size 1 is supported for now.

    Args:
        model: The model to use for computing logits
        prompt_ids: Prompt token IDs
        gen_ids: Generated token IDs
        device: Device to run computation on

    Returns:
        Per-token log probabilities for the generated tokens
    """
    token_ids = torch.tensor(prompt_ids + gen_ids, dtype=torch.long, device=device)
    prompt_len = len(prompt_ids)
    gen_len = len(gen_ids)
    attention_masks = build_varlen_metadata([(token_ids, prompt_len, gen_len)], device)

    full_tensor = token_ids.unsqueeze(0)

    # NOTE: We should move towards batching to improve efficiency here
    # See https://github.com/pytorch/torchtitan/issues/2674
    # Explicit positions avoid dynamic rope_cache[0:seqlen] slice in RoPE,
    # which breaks torch.compile with symbolic shapes.
    seq_len = full_tensor.shape[1]
    positions = torch.arange(seq_len, device=device).unsqueeze(0)

    logits = model(full_tensor, attention_masks=attention_masks, positions=positions)

    # Convert to float32 for numerical stability
    logits_f32 = logits[:, :-1, :].to(torch.float32)
    log_probs = F.log_softmax(logits_f32, dim=-1)
    target_tokens = full_tensor[:, 1:]

    # Extract log probs for generated tokens only
    gen_start_idx = prompt_len - 1
    gen_end_idx = gen_start_idx + gen_len

    gen_token_logprobs = log_probs[0, gen_start_idx:gen_end_idx, :]
    gen_token_ids_tensor = target_tokens[0, gen_start_idx:gen_end_idx]
    token_lps = gen_token_logprobs.gather(
        1, gen_token_ids_tensor.unsqueeze(-1)
    ).squeeze(-1)

    return token_lps


def compute_policy_gradient_loss(
    model: torch.nn.Module,
    vllm_token_ids: list[list[int]],
    prompt_token_ids: list[list[int]],
    advantages: torch.Tensor,
    ref_token_log_probs: list[torch.Tensor],
    kl_coef: float = 0.1,
    ppo_clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    log_probs_fn: callable = compute_token_log_probs,
) -> tuple[torch.Tensor, dict, list[torch.Tensor]]:
    """
    Compute GRPO/PPO policy gradient loss with per-token KL divergence.

    Uses per-token log ratios (averaged across tokens) instead of per-sequence
    sums to prevent ratio explosion when sequences are long.

    Args:
        model: Current policy model
        vllm_token_ids: Generated token IDs for each completion
        prompt_token_ids: Prompt token IDs for each completion
        advantages: [batch] - Advantages for each sample
        ref_token_log_probs: Per-token log probs from reference model (frozen)
        kl_coef: KL divergence penalty coefficient
        ppo_clip_eps: PPO clipping epsilon
        entropy_coef: Entropy bonus coefficient

    Returns:
        loss: Total loss (PG + entropy + KL)
        metrics: Training metrics dict
        batch_token_log_probs: List of per-token log probs for each sample (for verification)
    """
    device = next(model.parameters()).device
    advantages = advantages.to(device)

    # Compute per-token log probs under current policy (WITH GRADIENTS)
    batch_token_log_probs = []

    for prompt_toks, gen_toks in zip(prompt_token_ids, vllm_token_ids):
        token_lps = log_probs_fn(
            model,
            prompt_toks,
            gen_toks,
            device,
        )
        batch_token_log_probs.append(token_lps)

    # Per-token log ratios and KL, averaged across tokens per sample
    per_sample_mean_log_ratio = []
    per_sample_mean_kl = []
    all_token_log_probs = []

    for policy_token_lps, ref_token_lps in zip(
        batch_token_log_probs, ref_token_log_probs
    ):
        # Per-token log ratio: log(pi/pi_ref) for each token
        token_log_ratio = policy_token_lps - ref_token_lps.detach()
        # Average across tokens in this sequence
        per_sample_mean_log_ratio.append(token_log_ratio.mean())
        # Per-token KL: E[ratio - 1 - log_ratio] (Schulman approx)
        token_ratio = torch.exp(token_log_ratio)
        token_kl = token_ratio - 1 - token_log_ratio
        per_sample_mean_kl.append(token_kl.mean())
        all_token_log_probs.append(policy_token_lps)

    mean_log_ratio = torch.stack(per_sample_mean_log_ratio)  # [batch]
    mean_kl = torch.stack(per_sample_mean_kl)  # [batch]

    # PPO clipped objective using per-token-averaged ratio
    ratio = torch.exp(mean_log_ratio)
    unclipped_loss = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1 - ppo_clip_eps, 1 + ppo_clip_eps)
    clipped_loss = clipped_ratio * advantages
    pg_loss = -torch.min(unclipped_loss, clipped_loss).mean()

    # Entropy bonus (averaged across all tokens)
    all_token_lps = torch.cat(all_token_log_probs)
    entropy = -all_token_lps.mean()
    entropy_bonus = -entropy_coef * entropy

    # KL divergence penalty (averaged across samples)
    kl_div = mean_kl.mean()

    # Total loss
    total_loss = pg_loss + entropy_bonus + kl_coef * kl_div

    metrics = {
        "pg_loss": pg_loss.item(),
        "entropy": entropy.item(),
        "kl_div": kl_div.item(),
        "ratio_mean": ratio.mean().item(),
        "ratio_clipped_frac": (torch.abs(ratio - clipped_ratio) > 1e-6)
        .float()
        .mean()
        .item(),
    }

    return total_loss, metrics, batch_token_log_probs


def verify_logprob_identity(
    vllm_token_log_probs: list[list[float]],
    batch_token_log_probs: list[torch.Tensor],
) -> dict:
    """
    Check if vLLM log probs and computed log probs are bit-wise identical,
    and compute the log ratio (train/generator) between them.

    Args:
        vllm_token_log_probs: Per-token log probs from vLLM (generator)
        batch_token_log_probs: Per-token log probs computed by the trainer model

    Returns:
        Verification result dict with identity status, delta info, and log ratio stats
    """
    result = {
        "logprob_bitwise_identical": True,
        "num_samples_checked": len(vllm_token_log_probs),
        "total_tokens_checked": 0,
        "num_tokens_different": 0,
        "logprob_max_delta": 0.0,
        "avg_delta": 0.0,
        "logprob_diff_mean": 0.0,
        "logprob_diff_max": 0.0,
    }

    all_deltas = []
    all_log_ratios = []

    for vllm_lps, titan_lps in zip(vllm_token_log_probs, batch_token_log_probs):
        # Convert vLLM log probs to tensor
        vllm_tensor = torch.tensor(vllm_lps, dtype=torch.float32)
        # Convert titan log probs to float32 for comparison
        titan_tensor = titan_lps.detach().cpu().float()

        num_tokens = len(vllm_lps)
        result["total_tokens_checked"] += num_tokens

        # Check bitwise identity
        bitwise_match = torch.equal(vllm_tensor, titan_tensor)

        if not bitwise_match:
            result["logprob_bitwise_identical"] = False
            num_different = (vllm_tensor != titan_tensor).sum().item()
            result["num_tokens_different"] += num_different
            deltas = (vllm_tensor - titan_tensor).abs()
            all_deltas.append(deltas)

        # Log ratio: log(pi_train / pi_generator) = logprob_train - logprob_generator
        # Should be 0 when weights are identical (ratio = 1)
        all_log_ratios.append(titan_tensor - vllm_tensor)

    # Compute aggregate delta stats
    if all_deltas:
        combined_deltas = torch.cat(all_deltas)
        result["logprob_max_delta"] = combined_deltas.max().item()
        result["avg_delta"] = combined_deltas.mean().item()

    # Compute log ratio stats
    if all_log_ratios:
        combined_log_ratios = torch.cat(all_log_ratios)
        result["logprob_diff_mean"] = combined_log_ratios.mean().item()
        result["logprob_diff_max"] = combined_log_ratios.abs().max().item()

    return result
