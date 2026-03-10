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
) -> torch.Tensor:
    """
    Apply temperature, top-k, top-p filtering and sample one token.

    Parameters
    ----------
    logits : Tensor, shape (1, vocab_size)
    temperature : float
    top_k : int
    top_p : float

    Returns
    -------
    LongTensor, shape (1, 1)
    """
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k > 0:
        topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        logits = logits.masked_fill(logits < topk_vals[:, -1:], float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = (cumulative_probs - F.softmax(sorted_logits, dim=-1)) >= top_p
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_stream(
    model: TransformerLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_id: int | None = None,
) -> Iterator[int]:
    """
    Autoregressive generation yielding one token id at a time.

    Usage:
        for token_id in generate_stream(model, prompt, ...):
            print(tokenizer.decode([token_id]), end="", flush=True)

    Parameters
    ----------
    model : TransformerLM
    prompt_ids : LongTensor, shape (1, prompt_len)
    max_new_tokens : int
    temperature : float
    top_k : int
    top_p : float
    eos_id : int or None

    Yields
    ------
    int — one token id per step.
    """
    model.eval()
    seq = prompt_ids.clone()
    max_seq_len = model.config.max_seq_len

    for _ in range(max_new_tokens):
        input_ids = seq[:, -max_seq_len:] if seq.size(1) > max_seq_len else seq
        logits = model(input_ids)[:, -1, :]

        next_token = _sample_next_token(logits, temperature, top_k, top_p)
        token_id = next_token.item()

        seq = torch.cat([seq, next_token], dim=1)
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
) -> torch.Tensor:
    """
    Autoregressive generation returning the full sequence.

    Parameters
    ----------
    model : TransformerLM
    prompt_ids : LongTensor, shape (1, prompt_len)
    max_new_tokens, temperature, top_k, top_p, eos_id : see generate_stream

    Returns
    -------
    LongTensor, shape (1, prompt_len + generated_len)
    """
    generated = list(generate_stream(model, prompt_ids, max_new_tokens, temperature, top_k, top_p, eos_id))
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
) -> torch.Tensor:
    """
    Beam search decoding. Deterministic.

    Maintains beam_size hypotheses. At each step, expands each with top-k
    tokens, scores them, keeps top beam_size.
    Score = log_prob / length^alpha.

    Parameters
    ----------
    model : TransformerLM
    prompt_ids : LongTensor, shape (1, prompt_len)
    max_new_tokens : int
    beam_size : int
    length_penalty : float — 0=no penalty, 1=full normalization.
    eos_id : int or None

    Returns
    -------
    LongTensor, shape (1, seq_len)
    """
    model.eval()
    max_seq_len = model.config.max_seq_len
    prompt_len = prompt_ids.size(1)

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
            logits = model(input_ids)[:, -1, :]
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
