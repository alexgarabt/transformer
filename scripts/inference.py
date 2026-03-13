"""
Generate text from a trained model.

Loads params.json to reconstruct model + tokenizer automatically.
Supports streaming output, sampling, greedy, and beam search.

Usage:
    # Sampling (params.json auto-detected from checkpoint dir)
    uv run python scripts/inference.py --checkpoint checkpoints/llama_124m/best_model.pt --prompt "Once upon a time"

    # Explicit params.json path
    uv run python scripts/inference.py --checkpoint checkpoints/llama_124m/best_model.pt --params checkpoints/llama_124m/params.json --prompt "Hello"

    # Greedy
    uv run python scripts/inference.py --checkpoint checkpoints/llama_124m/best_model.pt --prompt "The answer is" --strategy greedy

    # Beam search
    uv run python scripts/inference.py --checkpoint checkpoints/llama_124m/best_model.pt --prompt "In the beginning" --strategy beam --beam-size 5

    # Full sampling control
    uv run python scripts/inference.py --checkpoint checkpoints/llama_124m/best_model.pt --prompt "Hello" --temperature 0.8 --top-k 50 --top-p 0.95 --repetition-penalty 1.3

    # No streaming
    uv run python scripts/inference.py --checkpoint checkpoints/llama_124m/best_model.pt --prompt "Test" --no-stream
"""

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import torch

from transformer.config import TransformerLMConfig
from transformer.models import TransformerLM
from transformer.data import Tokenizer
from transformer.generation import generate, generate_stream, beam_search


def load_model_and_tokenizer(
    checkpoint_path: str,
    params_path: str | None,
    device: str,
) -> tuple[TransformerLM, Tokenizer, TransformerLMConfig]:
    """
    Reconstruct model and tokenizer from checkpoint + params.json.

    If params_path is None, looks for params.json in the same directory
    as the checkpoint file.
    """
    ckpt_path = Path(checkpoint_path)

    # Find params.json: explicit path > same dir as checkpoint
    if params_path is not None:
        p_path = Path(params_path)
    else:
        p_path = ckpt_path.parent / "params.json"

    if not p_path.exists():
        raise FileNotFoundError(
            f"params.json not found at {p_path}. "
            f"Pass --params /path/to/params.json explicitly."
        )

    with open(p_path) as f:
        params = json.load(f)

    # Reconstruct model config
    valid_fields = {f.name for f in fields(TransformerLMConfig)}
    config = TransformerLMConfig(**{k: v for k, v in params["model"].items() if k in valid_fields})

    # Load model weights
    model = TransformerLM(config)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Handle torch.compile prefix
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = Tokenizer(params["tokenizer_path"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {n_params:,} params, vocab={config.vocab_size}, d_model={config.d_model}, "
          f"layers={config.n_layers}, heads={config.n_heads}")
    print(f"Tokenizer: {params['tokenizer_path']} ({tokenizer.vocab_size} tokens)")
    print(f"Checkpoint: {ckpt_path} (epoch {checkpoint.get('epoch', '?')}, loss {checkpoint.get('loss', '?'):.4f})")
    print(f"Params: {p_path}")

    return model, tokenizer, config


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--params", type=str, default=None, help="Path to params.json (default: same dir as checkpoint)")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt (empty = generate from BOS)")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--strategy", type=str, default="sample", choices=["sample", "greedy", "beam"])
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0, help="0 = disabled")
    parser.add_argument("--top-p", type=float, default=0.95, help="1.0 = disabled")
    parser.add_argument("--repetition-penalty", type=float, default=1.2, help="1.0 = disabled")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # ── Load model ──
    model, tokenizer, config = load_model_and_tokenizer(args.checkpoint, args.params, args.device)

    # ── Tokenize prompt ──
    if args.prompt:
        prompt_ids = tokenizer.encode(args.prompt, add_bos=True)
    else:
        prompt_ids = [tokenizer.bos_id]
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)

    print(f"\nPrompt ({len(prompt_ids)} tokens): {args.prompt!r}")
    if args.strategy == "sample":
        print(f"Strategy: sample (temp={args.temperature}, top_k={args.top_k}, "
              f"top_p={args.top_p}, rep_penalty={args.repetition_penalty})")
    elif args.strategy == "beam":
        print(f"Strategy: beam (size={args.beam_size}, len_penalty={args.length_penalty}, "
              f"rep_penalty={args.repetition_penalty})")
    else:
        print(f"Strategy: greedy (rep_penalty={args.repetition_penalty})")
    print("─" * 60)

    # ── Print prompt ──
    if args.prompt:
        print(args.prompt, end="", flush=True)

    # ── Beam search ──
    if args.strategy == "beam":
        output_ids = beam_search(
            model, prompt_tensor,
            max_new_tokens=args.max_tokens,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            eos_id=tokenizer.eos_id,
            repetition_penalty=args.repetition_penalty,
        )
        full_text = tokenizer.decode(output_ids[0].tolist())
        prompt_text = tokenizer.decode(prompt_ids)
        print(full_text[len(prompt_text):])
        n_generated = output_ids.size(1) - len(prompt_ids)

    # ── Greedy / non-streaming sample ──
    elif args.no_stream or args.strategy == "greedy":
        output_ids = generate(
            model, prompt_tensor,
            max_new_tokens=args.max_tokens,
            temperature=0.0 if args.strategy == "greedy" else args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_id=tokenizer.eos_id,
            repetition_penalty=args.repetition_penalty,
        )
        full_text = tokenizer.decode(output_ids[0].tolist())
        prompt_text = tokenizer.decode(prompt_ids)
        print(full_text[len(prompt_text):])
        n_generated = output_ids.size(1) - len(prompt_ids)

    # ── Streaming sample ──
    else:
        n_generated = 0
        all_ids = list(prompt_ids)
        prompt_text = tokenizer.decode(prompt_ids)
        prev_text = prompt_text

        for token_id in generate_stream(
            model, prompt_tensor,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_id=tokenizer.eos_id,
            repetition_penalty=args.repetition_penalty,
        ):
            all_ids.append(token_id)
            n_generated += 1

            full_text = tokenizer.decode(all_ids)
            new_chars = full_text[len(prev_text):]
            if new_chars:
                sys.stdout.write(new_chars)
                sys.stdout.flush()
                prev_text = full_text

        print()

    print(f"\n─── Generated {n_generated} tokens ───")


if __name__ == "__main__":
    main()
