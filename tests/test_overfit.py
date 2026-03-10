import torch
import torch.nn.functional as F

from transformer import TransformerLM, TransformerEncoderDecoder
from transformer.config import TransformerLMConfig, EncoderDecoderConfig

VOCAB_SIZE = 100
D_MODEL = 64
N_HEADS = 4
D_FF = 128
SEQ_LEN = 16
BATCH_SIZE = 2
N_LAYERS = 2


def test_lm_overfit_single_batch():
    torch.manual_seed(42)

    config = TransformerLMConfig(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ff=D_FF, max_seq_len=SEQ_LEN, dropout=0.0,
    )
    model = TransformerLM(config)
    model.train()

    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    for _ in range(200):
        logits = model(input_ids)
        # Next-token prediction: predict token t+1 from position t
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, VOCAB_SIZE),
            input_ids[:, 1:].reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.1, f"LM failed to overfit: loss={loss.item():.4f}"


def test_encoder_decoder_overfit():
    torch.manual_seed(42)

    config = EncoderDecoderConfig(
        src_vocab_size=VOCAB_SIZE, tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL, n_heads=N_HEADS,
        encoder_layers=N_LAYERS, decoder_layers=N_LAYERS,
        d_ff=D_FF, max_seq_len=SEQ_LEN, dropout=0.0,
    )
    model = TransformerEncoderDecoder(config)
    model.train()

    src_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    tgt_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(100):
        logits = model(src_ids, tgt_ids)
        # Teacher-forced next-token: predict tgt[t+1] from tgt[t]
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, VOCAB_SIZE),
            tgt_ids[:, 1:].reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.5, f"Enc-Dec failed to overfit: loss={loss.item():.4f}"
