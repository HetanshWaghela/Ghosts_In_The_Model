"""
Model utility functions for machine unlearning research.

Key point:
- Many targets (e.g., "New York") are multi-token under GPT-2 BPE.
- For research claims, it's safer to evaluate the probability of the *full target string*,
  not just the first token.

This module provides:
- first-token probability
- sequence probability (geometric mean per-token probability under teacher forcing)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


TargetProbMode = Literal["first_token", "sequence_geomean"]


@dataclass(frozen=True)
class TargetProbs:
    """
    Target probability statistics for a single (prompt, target) pair.

    - first_token_prob: P(t0 | prompt)
    - seq_logprob_sum: log P(target_tokens | prompt) = Σ log P(t_i | prompt, t_<i)
    - seq_logprob_avg: mean log-prob per target token
    - seq_prob_geomean: exp(seq_logprob_avg)  (geometric mean probability per token)
    """

    target_token_ids: List[int]
    first_token_prob: float
    seq_logprob_sum: float
    seq_logprob_avg: float
    seq_prob_geomean: float


def _encode_target(tokenizer: GPT2Tokenizer, target: str) -> List[int]:
    # Prepend a space because GPT-2 BPE is space-sensitive.
    return tokenizer.encode(" " + target, add_special_tokens=False)


@torch.no_grad()
def get_target_probs(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    target: str,
    device: str,
) -> TargetProbs:
    """
    Compute both first-token and sequence-level target probabilities.
    """
    target_ids = _encode_target(tokenizer, target)
    if not target_ids:
        return TargetProbs(
            target_token_ids=[],
            first_token_prob=0.0,
            seq_logprob_sum=float("-inf"),
            seq_logprob_avg=float("-inf"),
            seq_prob_geomean=0.0,
        )

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        # Edge case: empty prompt; treat as BOS-less sequence.
        prompt_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [0]

    input_ids = torch.tensor([prompt_ids + target_ids], device=device)
    attention_mask = torch.ones_like(input_ids)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[0]  # [seq_len, vocab]

    start = len(prompt_ids) - 1
    positions = torch.arange(start, start + len(target_ids), device=device)
    logits_sel = logits.index_select(0, positions)  # [target_len, vocab]

    log_probs = torch.log_softmax(logits_sel, dim=-1)
    target_tensor = torch.tensor(target_ids, device=device).unsqueeze(1)  # [target_len, 1]
    token_log_probs = log_probs.gather(1, target_tensor).squeeze(1)  # [target_len]

    seq_logprob_sum = float(token_log_probs.sum().item())
    seq_logprob_avg = float(token_log_probs.mean().item())
    first_token_prob = float(token_log_probs[0].exp().item())
    seq_prob_geomean = float(torch.exp(token_log_probs.mean()).item())

    return TargetProbs(
        target_token_ids=target_ids,
        first_token_prob=first_token_prob,
        seq_logprob_sum=seq_logprob_sum,
        seq_logprob_avg=seq_logprob_avg,
        seq_prob_geomean=seq_prob_geomean,
    )


@torch.no_grad()
def get_target_probability(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    target: str,
    device: str,
    *,
    mode: TargetProbMode = "sequence_geomean",
) -> float:
    """
    Convenience wrapper returning a single scalar probability.

    Default is sequence_geomean to be robust to multi-token targets.
    """
    probs = get_target_probs(model, tokenizer, prompt, target, device)
    if mode == "first_token":
        return probs.first_token_prob
    return probs.seq_prob_geomean