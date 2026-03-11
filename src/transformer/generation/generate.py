"""
Text generation with autoregressive decoding.

Strategies:
    temperature=0       → greedy (argmax)
    temperature>0       → sampling from softmax(logits/T)
    top_k > 0           → filter to top-k tokens before sampling
    top_p < 1.0         → nucleus sampling
    beam_size > 1       → beam search (deterministic)

generate()          → returns full sequence as tensor
generate_stream()   → yields token ids one by one (for live output)
beam_search()       → returns best sequence from beam search

All sampling strategies support repetition_penalty to discourage loops.
"""

import torch
import torch.nn.functional as F
from collections.abc import Iterator
from ..models.lm import TransformerLM


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    generated_ids: list[int] | None = None,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """
    Apply repetition penalty, temperature, top-k, top-p filtering, then sample.

    Parameters
    ----------
    logits : Tensor, shape (1, vocab_size)
    temperature : float — 0 = greedy.
    top_k : int — 0 = disabled.
    top_p : float — 1.0 = disabled.
    generated_ids : list of already generated token ids (for repetition penalty).
    repetition_penalty : float — 1.0 = disabled. >1.0 penalizes repetition.

    Returns
    -------
    LongTensor, shape (1, 1)
    """
    # Greedy
    if temperature <= 0:
        if repetition_penalty != 1.0 and generated_ids:
            logits = _apply_repetition_penalty(logits, generated_ids, repetition_penalty)
        return logits.argmax(dim=-1, keepdim=True)

    # Repetition penalty
    if repetition_penalty != 1.0 and generated_ids:
        logits = _apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    # Temperature
    logits = logits / temperature

    # Top-k
    if top_k > 0:
        topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        logits = logits.masked_fill(logits < topk_vals[:, -1:], float("-inf"))

    # Top-p (nucleus)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = (cumulative_probs - F.softmax(sorted_logits, dim=-1)) >= top_p
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: list[int],
    penalty: float,
) -> torch.Tensor:
    """
    Penalize tokens that have already been generated.

    For each previously generated token:
        if logit > 0: divide by penalty (makes it less attractive)
        if logit < 0: multiply by penalty (makes it more negative)
    """
    logits = logits.clone()
    token_set = torch.tensor(list(set(generated_ids)), device=logits.device)
    scores = logits[:, token_set]
    scores = torch.where(scores > 0, scores / penalty, scores * penalty)
    logits[:, token_set] = scores
    return logits


@torch.no_grad()
def generate_stream(
    model: TransformerLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_id: int | None = None,
    repetition_penalty: float = 1.0,
) -> Iterator[int]:
    """
    Autoregressive generation yielding one token id at a time.

    Parameters
    ----------
    model : TransformerLM
    prompt_ids : LongTensor, shape (1, prompt_len)
    max_new_tokens : int
    temperature : float
    top_k : int
    top_p : float
    eos_id : int or None
    repetition_penalty : float — 1.0 = disabled. 1.2-1.5 typical for small models.

    Yields
    ------
    int — one token id per step.
    """
    model.eval()
    seq = prompt_ids.clone()
    max_seq_len = model.config.max_seq_len
    generated: list[int] = []

    for _ in range(max_new_tokens):
        input_ids = seq[:, -max_seq_len:] if seq.size(1) > max_seq_len else seq
        logits = model(input_ids)[:, -1, :]

        next_token = _sample_next_token(logits, temperature, top_k, top_p, generated, repetition_penalty)
        token_id = next_token.item()

        seq = torch.cat([seq, next_token], dim=1)
        generated.append(token_id)
        yield token_id

        if eos_id is not None and token_id == eos_id:
            return


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_id: int | None = None,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """
    Autoregressive generation returning the full sequence.

    Returns
    -------
    LongTensor, shape (1, prompt_len + generated_len)
    """
    generated = list(generate_stream(
        model, prompt_ids, max_new_tokens, temperature,
        top_k, top_p, eos_id, repetition_penalty,
    ))
    if not generated:
        return prompt_ids
    new_tokens = torch.tensor([generated], dtype=torch.long, device=prompt_ids.device)
    return torch.cat([prompt_ids, new_tokens], dim=1)


@torch.no_grad()
def beam_search(
    model: TransformerLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 200,
    beam_size: int = 5,
    length_penalty: float = 0.6,
    eos_id: int | None = None,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """
    Beam search decoding with optional repetition penalty. Deterministic.

    Returns
    -------
    LongTensor, shape (1, seq_len)
    """
    model.eval()
    max_seq_len = model.config.max_seq_len
    prompt_len = prompt_ids.size(1)

    # Each beam: (sequence, cumulative_log_prob, finished)
    beams: list[tuple[torch.Tensor, float, bool]] = [(prompt_ids.clone(), 0.0, False)]

    def score_beam(candidate: tuple[torch.Tensor, float, bool]) -> float:
        seq, log_prob, _ = candidate
        gen_len = seq.size(1) - prompt_len
        if gen_len == 0:
            return log_prob
        return log_prob / (gen_len ** length_penalty)

    for _ in range(max_new_tokens):
        candidates: list[tuple[torch.Tensor, float, bool]] = []

        for seq, log_prob, finished in beams:
            if finished:
                candidates.append((seq, log_prob, True))
                continue

            input_ids = seq[:, -max_seq_len:] if seq.size(1) > max_seq_len else seq
            logits = model(input_ids)[:, -1, :]  # (1, vocab_size)

            # Apply repetition penalty to beam's generated tokens
            if repetition_penalty != 1.0:
                gen_ids = seq[0, prompt_len:].tolist()
                if gen_ids:
                    logits = _apply_repetition_penalty(logits, gen_ids, repetition_penalty)

            log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
            topk_lps, topk_ids = torch.topk(log_probs, beam_size)

            for i in range(beam_size):
                token = topk_ids[i].unsqueeze(0).unsqueeze(0)
                new_seq = torch.cat([seq, token], dim=1)
                new_lp = log_prob + topk_lps[i].item()
                is_eos = eos_id is not None and topk_ids[i].item() == eos_id
                candidates.append((new_seq, new_lp, is_eos))

        candidates.sort(key=score_beam, reverse=True)
        beams = candidates[:beam_size]

        if all(fin for _, _, fin in beams):
            break

    best_seq, _, _ = max(beams, key=score_beam)
    return best_seq
