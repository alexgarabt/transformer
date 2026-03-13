---
language:
  - en
license: mit
tags:
  - transformer
  - language-model
  - pytorch
  - from-scratch
  - llama
  - decoder-only
  - causal-lm
library_name: pytorch
pipeline_tag: text-generation
datasets:
  - HuggingFaceFW/fineweb-edu
model-index:
  - name: llama-124m-fineweb
    results:
      - task:
          type: text-generation
          name: Language Modeling
        dataset:
          name: FineWeb-Edu (sample-10BT)
          type: HuggingFaceFW/fineweb-edu
          config: sample-10BT
        metrics:
          - name: Validation Loss
            type: loss
            value: 3.1626
          - name: Perplexity
            type: perplexity
            value: 23.62
---

# LLaMA-124M FineWeb-Edu

A **124M parameter LLaMA-style decoder-only transformer** trained from scratch on [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (sample-10BT, ~9.65 billion tokens).

Every component — attention, feed-forward layers, normalization, positional encoding, training loop — is implemented from scratch in pure PyTorch. No HuggingFace `transformers` library.

**Source code:** [github.com/alexgarabt/transformer](https://github.com/alexgarabt/transformer)

https://huggingface.co/alexgara/llama-124m/resolve/main/img/llama_124m.mp4

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Parameters | 124,472,064 |
| `d_model` | 768 |
| `n_heads` | 12 |
| `n_layers` | 12 |
| `d_ff` | 2,560 |
| `max_seq_len` | 1,024 |
| Activation | SwiGLU |
| Normalization | RMSNorm (pre-norm) |
| Position encoding | Learned |
| Weight tying | Yes (embedding ↔ LM head) |
| Vocab size | 32,000 (SentencePiece BPE) |
| Dropout | 0.0 |
| Bias | No |

## Training Details

| Detail | Value |
|--------|-------|
| Dataset | FineWeb-Edu sample-10BT (~9.65B tokens) |
| Epochs | 1 |
| Effective batch size | 128 × 1,024 × 4 = **524,288 tokens/step** |
| Optimizer | AdamW (lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1) |
| LR schedule | Cosine decay with 750 warmup steps |
| Precision | bfloat16 (mixed precision) |
| Attention | Flash Attention (`F.scaled_dot_product_attention`) |
| Gradient clipping | 1.0 (max norm) |
| Hardware | 1× NVIDIA H200 (~110 GB VRAM) |
| Training time | ~6 hours |
| Estimated cost | ~$24 |
| Final val loss | **3.1626** |
| Final perplexity | **~23.6** |
| Seed | 42 |

## Usage

### Quick Start

```bash
git clone https://github.com/alexgarabt/transformer
cd transformer
uv sync --extra data
uv run python scripts/inference.py --prompt "The meaning of life" --device cpu
```

### Download Weights Manually

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download("alexgarabt/llama-124m-fineweb", "model.pt")
params_path = hf_hub_download("alexgarabt/llama-124m-fineweb", "params.json")
tokenizer_path = hf_hub_download("alexgarabt/llama-124m-fineweb", "tokenizer.model")
```

### Load and Generate (Python)

```python
import torch
from transformer.config import TransformerLMConfig
from transformer.models import TransformerLM
from transformer.data import Tokenizer
from transformer.generation import generate

# Load config
import json
with open(params_path) as f:
    params = json.load(f)

config = TransformerLMConfig(**params["model"])
model = TransformerLM(config)

# Load weights
checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load tokenizer and generate
tokenizer = Tokenizer(tokenizer_path)
prompt_ids = torch.tensor([tokenizer.encode("Once upon a time", add_bos=True)])
output = generate(model, prompt_ids, max_new_tokens=100, temperature=0.8, top_p=0.95)
print(tokenizer.decode(output[0].tolist()))
```

## Files in This Repository

| File | Description | Size |
|------|-------------|------|
| `model.pt` | Model weights (clean state dict) | ~500 MB |
| `params.json` | Model & training configuration | ~1 KB |
| `tokenizer.model` | SentencePiece BPE tokenizer | ~786 KB |
| `tokenizer.vocab` | Vocabulary file | ~503 KB |
| `config/llama_124M.json` | Training configuration | ~1 KB |
| `runs/` | TensorBoard training logs | ~95 MB |

## Limitations

- **Base model only** — no instruction tuning (SFT), no RLHF, no DPO. The model generates coherent English text but does not follow instructions or answer questions reliably.
- **Small model** — 124M parameters is relatively small. Generation quality is limited compared to larger models.
- **Single epoch** — trained for only one pass over the dataset.
- **English only** — the tokenizer and training data are English.
- **Max 1024 tokens** — the model's positional encoding supports sequences up to 1024 tokens.

