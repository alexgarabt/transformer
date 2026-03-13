# Transformer

A complete Transformer implementation **from scratch** in PyTorch. Every component — multi-head attention, feed-forward networks, normalization layers, positional encoding, training loop, text generation — is hand-built. No HuggingFace `transformers` library.

Includes a **124M parameter pretrained model** trained on FineWeb-Edu (10B tokens), available on [HuggingFace Hub](https://huggingface.co/alexgara/llama-124m).

https://github.com/user-attachments/assets/c140082e-55e8-4664-94bc-1feb06ecfc8f

## Quick Start

```bash
git clone https://github.com/alexgarabt/transformer
cd transformer
uv sync --extra data

# Generate text using the pretrained model (downloads automatically)
uv run python scripts/inference.py --prompt "Once upon a time" --device cpu
```

## Pretrained Model

| | |
|---|---|
| **Parameters** | 124,472,064 |
| **Architecture** | Decoder-only (LLaMA-style) |
| **Layers / Heads / d_model** | 12 / 12 / 768 |
| **Feed-forward dim** | 2,560 (SwiGLU) |
| **Vocab / Seq len** | 32,000 / 1,024 |
| **Training data** | FineWeb-Edu sample-10BT (~9.65B tokens) |
| **Training time** | ~6 hours on 1x NVIDIA H200 |
| **Val loss / Perplexity** | 3.1626 / ~23.6 |
| **HuggingFace** | [alexgarabt/llama-124m-fineweb](https://huggingface.co/alexgara/llama-124m) |

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/alexgarabt/transformer
cd transformer

# Core (training + inference with local checkpoints)
uv sync

# With HuggingFace Hub support (download/upload models)
uv sync --extra data

# With dev tools (pytest, ipython, jupyter)
uv sync --extra dev --extra data
```

## Project Structure

```
src/transformer/
    config.py              TransformerLMConfig, TrainingConfig, ...
    models/
        lm.py              TransformerLM (decoder-only, GPT/LLaMA style)
        encoder_decoder.py TransformerEncoderDecoder (seq2seq)
        masked_lm.py       MaskedLM (BERT style)
        decoder.py          TransformerDecoder
        encoder.py          TransformerEncoder
    layers/
        attention.py       MultiHeadAttention (Flash Attention)
        block.py           TransformerBlock (causal + cross-attention)
        embedding.py       Token + positional embeddings
        feedforward.py     FeedForward + SwiGLU
        norm.py            LayerNorm + RMSNorm
    data/
        dataset.py         TextDataset (mmap + parallel tokenization)
        tokenizer.py       SentencePiece BPE wrapper
    training/
        trainer.py         Trainer (TensorBoard, mixed precision, grad accum)
        metrics.py         Loss, perplexity, attention analysis
        utils.py           Cosine schedule, weight init, seed
    generation/
        generate.py        Sampling, beam search, streaming

scripts/
    train_llm.py           Train tokenizer, prepare data, train, continue
    inference.py           Generate text (local or from HuggingFace Hub)
    hub.py                 Upload/download model to/from HuggingFace Hub
    download_data.py       Download FineWeb-Edu and other datasets

config/
    llama_124M.json        124M model config (SwiGLU, RMSNorm, 12 layers)
    gpt2_small_test.json   17M test model config (GELU, LayerNorm, 6 layers)

tests/                     34 tests covering all components
```

## Architectures

### TransformerLM (Decoder-Only)

The main model. GPT/LLaMA-style causal language model:

```
Input IDs → Embedding + Positional → N × [RMSNorm → Causal Self-Attention → RMSNorm → SwiGLU FFN] → RMSNorm → LM Head → Logits
```

- Causal masking: each token only attends to previous tokens
- Weight tying: embedding and LM head share the same weight matrix
- Pre-norm architecture (norm before attention/FFN, not after)

### TransformerEncoderDecoder (Seq2Seq)

For translation and sequence-to-sequence tasks:

- Encoder: bidirectional self-attention over source sequence
- Decoder: causal self-attention + cross-attention to encoder output
- Separate source/target vocabularies (optionally shared)

### MaskedLM (BERT-Style)

Encoder-only model for masked language modeling:

- Bidirectional attention (no causal mask)
- Predicts randomly masked tokens
- Pre-training objective for downstream fine-tuning

## Training Pipeline

### 1. Download data

```bash
# FineWeb-Edu 10B tokens (~20GB download, ~45GB text)
uv run python scripts/download_data.py --dataset 10BT --output-dir data/llama_124m

# Smaller datasets for testing
uv run python scripts/download_data.py --dataset tinystories --output-dir data/small_test
```

### 2. Train tokenizer

```bash
uv run python scripts/train_llm.py train-tokenizer \
    --input data/llama_124m/train.txt \
    --prefix data/llama_124m/tok_32k \
    --vocab-size 32000
```

### 3. Prepare tokenized data

```bash
uv run python scripts/train_llm.py prepare \
    --tokenizer data/llama_124m/tok_32k.model \
    --input data/llama_124m/train.txt \
    --output data/llama_124m/train_10bt.npy
```

### 4. Train

```bash
# From a config file
uv run python scripts/train_llm.py train --config config/llama_124M.json

# Continue training from a checkpoint
uv run python scripts/train_llm.py continue \
    --checkpoint checkpoints/llama_124m/weights.pt
```

### 5. Inference

```bash
# From HuggingFace Hub
uv run python scripts/inference.py --prompt "The meaning of life"

# From local checkpoint
uv run python scripts/inference.py \
    --checkpoint checkpoints/llama_124m/weights.pt \
    --prompt "Once upon a time" \
    --strategy sample --temperature 0.8 --top-p 0.95

# Beam search
uv run python scripts/inference.py \
    --checkpoint checkpoints/llama_124m/weights.pt \
    --prompt "In the beginning" \
    --strategy beam --beam-size 5

# Greedy decoding
uv run python scripts/inference.py \
    --checkpoint checkpoints/llama_124m/weights.pt \
    --prompt "Hello world" \
    --strategy greedy
```

### 6. Upload/Download

```bash
# Upload trained model to HuggingFace Hub
uv run python scripts/hub.py upload

# Download model for inference
uv run python scripts/hub.py download --output-dir hub_cache/
```

## Architecture Details

### Self-Attention

The core mechanism. For queries Q, keys K, values V with dimension d_k:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Multi-head attention runs h parallel attention heads, each with d_head = d_model / h dimensions, then concatenates and projects:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

Uses **Flash Attention** via `F.scaled_dot_product_attention` for efficient memory usage during training.

### SwiGLU Feed-Forward

The LLaMA-style gated feed-forward network:

$$\text{SwiGLU}(x) = (\text{Swish}(xW_1) \odot xW_3)W_2$$

where Swish(x) = x * sigmoid(x). Also supports standard ReLU and GELU activations.

### RMSNorm

Root Mean Square Layer Normalization (faster than LayerNorm, no mean centering):

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \epsilon}} \cdot \gamma$$

### Positional Encoding

Supports learned and sinusoidal positional encodings:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

## Training Features

- **Mixed precision**: bfloat16 (default), float16 with GradScaler, or float32
- **Gradient accumulation**: configurable steps for large effective batch sizes
- **Cosine LR schedule**: with linear warmup
- **Gradient clipping**: max norm clipping
- **TensorBoard logging**: loss, perplexity, learning rate, gradient norms, weight histograms, attention heatmaps
- **Checkpointing**: saves model + optimizer + scheduler state for resumption
- **torch.compile**: compatible (handles `_orig_mod.` prefix transparently)

## Models Trained

### LLaMA-124M (main)

- 124M params, 12 layers, 12 heads, SwiGLU + RMSNorm
- 1 epoch on FineWeb-Edu 10BT (~9.65B tokens)
- 1x NVIDIA H200, ~6 hours, bfloat16
- Val loss: 3.1626, perplexity ~23.6

### GPT2-Small Test (17M)

- ~17M params, 6 layers, 6 heads, GELU + LayerNorm
- Small dataset for rapid iteration and testing
- Trained locally on RTX 5080

## Tests

```bash
uv run pytest tests/ -v
```

34 tests covering attention, feed-forward, normalization, embeddings, transformer blocks, full models, and overfitting verification.

