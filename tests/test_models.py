import torch

from transformer import TransformerLM, TransformerEncoderDecoder, MaskedLM
from transformer.config import TransformerLMConfig
from conftest import D_MODEL, VOCAB_SIZE, BATCH_SIZE, SEQ_LEN


def test_transformer_lm_shape(lm_config, token_ids):
    model = TransformerLM(lm_config)
    logits = model(token_ids)
    assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)


def test_transformer_lm_weight_tying(lm_config):
    model = TransformerLM(lm_config)
    assert model.lm_head.weight is model.decoder.embedding.embedding.weight

    # Also verify disabling weight tying
    lm_config_no_tie = TransformerLMConfig(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=4,
        n_layers=2, d_ff=128, max_seq_len=SEQ_LEN,
        dropout=0.0, weight_tying=False,
    )
    model_no_tie = TransformerLM(lm_config_no_tie)
    assert model_no_tie.lm_head.weight is not model_no_tie.decoder.embedding.embedding.weight


def test_encoder_decoder_shape(enc_dec_config, token_ids):
    model = TransformerEncoderDecoder(enc_dec_config)
    src_ids = token_ids
    tgt_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    logits = model(src_ids, tgt_ids)
    assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)


def test_masked_lm_shape(masked_lm_config, token_ids):
    model = MaskedLM(masked_lm_config)
    logits = model(token_ids)
    assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)


def test_masked_lm_weight_tying(masked_lm_config):
    model = MaskedLM(masked_lm_config)
    assert model.lm_head.weight is model.encoder.embedding.embedding.weight
