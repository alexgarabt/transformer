"""
Generate text from a trained model.

Loads params.json to reconstruct model + tokenizer automatically.
Supports streaming output, sampling, greedy, and beam search.

Usage:
    # Sampling 
    uv run python scripts/inference.py --checkpoint checkpoints/best_model.pt --prompt "Once upon a time"

    # Greedy
    uv run python scripts/inference.py --checkpoint checkpoints/best_model.pt --prompt "The answer is" --strategy greedy

    # Beam search
    uv run python scripts/inference.py --checkpoint checkpoints/best_model.pt --prompt "In the beginning" --strategy beam --beam-size 5

    # Tune sampling
    uv run python scripts/inference.py --checkpoint checkpoints/best_model.pt --prompt "Hello" --temperature 0.6 --top-k 50 --top-p 0.9

    # Disable streaming
    uv run python scripts/inference.py --checkpoint checkpoints/best_model.pt --prompt "Test" --no-stream
"""

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import torch

from transformer.config import TransformerLMConfig
from transformer.models import TransformerLM
from transformer.data import Tokenizer
from transformer.generation import generate, generate_stream, beam_search


def load_model_and_tokenizer(checkpoint_path: str, device: str) -> tuple[TransformerLM, Tokenizer, TransformerLMConfig]:
    """
    Reconstruct model and tokenizer from checkpoint + params.json.

    Returns (model, tokenizer, config).
    """
    import json

    ckpt_path = Path(checkpoint_path)
    params_path = ckpt_path.parent / "params.json"

    if not params_path.exists():
        raise FileNotFoundError(f"params.json not found in {ckpt_path.parent}. Was the model trained with train_lm.py?")

    with open(params_path) as f:
        params = json.load(f)

    # Reconstruct config
    valid_fields = {f.name for f in fields(TransformerLMConfig)}
    config = TransformerLMConfig(**{k: v for k, v in params["model"].items() if k in valid_fields})

    # Load model
    model = TransformerLM(config)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = Tokenizer(params["tokenizer_path"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {n_params:,} params, vocab={config.vocab_size}, d_model={config.d_model}, "
          f"layers={config.n_layers}, heads={config.n_heads}")
    print(f"Tokenizer: {params['tokenizer_path']} ({tokenizer.vocab_size} tokens)")
    print(f"Checkpoint: {ckpt_path} (epoch {checkpoint.get('epoch', '?')}, loss {checkpoint.get('loss', '?'):.4f})")

    return model, tokenizer, config


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt (empty = generate from BOS)")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--strategy", type=str, default="sample", choices=["sample", "greedy", "beam"])
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0, help="0 = disabled")
    parser.add_argument("--top-p", type=float, default=0.95, help="1.0 = disabled")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming (print all at once)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # ── Load model ──
    model, tokenizer, config = load_model_and_tokenizer(args.checkpoint, args.device)

    # ── Tokenize prompt ──
    if args.prompt:
        prompt_ids = tokenizer.encode(args.prompt, add_bos=True)
    else:
        prompt_ids = [tokenizer.bos_id]
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)

    print(f"\nPrompt ({len(prompt_ids)} tokens): {args.prompt!r}")
    print(f"Strategy: {args.strategy}" + (f" (temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p})" if args.strategy == "sample" else ""))
    print("─" * 60)

    # ── Print prompt first ──
    if args.prompt:
        print(args.prompt, end="", flush=True)

    # ── Generate ──
    if args.strategy == "beam":
        output_ids = beam_search(
            model, prompt_tensor,
            max_new_tokens=args.max_tokens,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            eos_id=tokenizer.eos_id,
        )
        text = tokenizer.decode(output_ids[0].tolist())
        print(text)
        n_generated = output_ids.size(1) - len(prompt_ids)

    elif args.no_stream or args.strategy == "greedy":
        output_ids = generate(
            model, prompt_tensor,
            max_new_tokens=args.max_tokens,
            temperature=0.0 if args.strategy == "greedy" else args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_id=tokenizer.eos_id,
        )
        # Decode only the generated part
        generated_ids = output_ids[0, len(prompt_ids):].tolist()
        print(tokenizer.decode(generated_ids))
        n_generated = len(generated_ids)

    else:
        # Streaming: print each token as it's generated
        n_generated = 0
        token_buffer: list[int] = []

        for token_id in generate_stream(
            model, prompt_tensor,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_id=tokenizer.eos_id,
        ):
            token_buffer.append(token_id)
            n_generated += 1
            # Decode buffer to handle multi-token characters (UTF-8)
            text = tokenizer.decode(token_buffer)
            if text and not text.endswith("�"):  # wait if incomplete UTF-8
                sys.stdout.write(text)
                sys.stdout.flush()
                token_buffer.clear()

        # Flush remaining buffer
        if token_buffer:
            sys.stdout.write(tokenizer.decode(token_buffer))
            sys.stdout.flush()
        print()

    print(f"\n─── Generated {n_generated} tokens ───")


if __name__ == "__main__":
    main()
