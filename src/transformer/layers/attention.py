"""
Multi-Head Attention

Single class self-attention, cross-attention and causal masking.
    - Self attention:   forward(x, x, x)
    - Cross attention:  forward(x_dec, x_enc, x_enc)
    - Causal:           forward(x, x, x, causal=True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
     Multi-Head Attention (Vaswani et al., 2017).

    Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_head)) @ V

    Parameters
    ----------
    d_model : int
        Model dimension. Must be divisible by n_heads.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout on attention weights.
    bias : bool
        Whether to use bias in Q, K, V, O projections.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, bias: bool = False, **kwargs):
        super().__init__(**kwargs)
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        # single linear per projection -> all heads together
        self.W_Q = nn.Linear(d_model, d_model, bias)
        self.W_K = nn.Linear(d_model, d_model, bias)
        self.W_V = nn.Linear(d_model, d_model, bias)
        self.W_O = nn.Linear(d_model, d_model, bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q_src: torch.Tensor, k_src: torch.Tensor, v_src: torch.Tensor, mask: torch.Tensor | None = None, causal: bool = False, return_weights: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head attention.

        For self-attention, pass the same tensor for all three sources:
            attn(x, x, x)
        For cross-attention, queries come from one source, keys/values from another:
            attn(x_decoder, x_encoder, x_encoder)

        Parameters
        ----------
        q_src : Tensor, shape (batch, seq_q, d_model)
            Source for query projections.
        k_src : Tensor, shape (batch, seq_k, d_model)
            Source for key projections. seq_k may differ from seq_q in cross-attention.
        v_src : Tensor, shape (batch, seq_k, d_model)
            Source for value projections. Same sequence length as k_src.
        mask : Tensor or None, shape broadcastable to (batch, n_heads, seq_q, seq_k)
            Boolean mask. True positions are filled with -inf before softmax
            (e.g., padding mask to ignore pad tokens).
        causal : bool
            If True, applies an upper-triangular causal mask so that position i
            cannot attend to positions j > i.

        Returns
        -------
        Tensor, shape (batch, seq_q, d_model)
            Contextualized representations. Same shape as q_src.
        Or tuple of (output, attn_weights) if return_weights=True.
        """
        batch_size, seq_q, _ = q_src.shape
        seq_k = k_src.shape[1]

        # project and reshape into (batch, n_heads, seq, d_head)
        Q = self._reshape_to_heads(self.W_Q(q_src))
        K = self._reshape_to_heads(self.W_K(k_src))
        V = self._reshape_to_heads(self.W_V(v_src))

        # scaled dot product attention
        #scores = (Q @ K.transpose(-2, -1)) / self.scale

        ##########################################################
        ################# NORAMAL ATTENTION      #################
        ##########################################################
        ## apply masks
        #if causal:
        #    causal_mask = torch.triu(torch.ones(seq_q, seq_k, device=scores.device, dtype=torch.bool), diagonal=1)
        #    scores = scores.masked_fill(causal_mask, float("-inf"))

        #if mask is not None:
        #    scores = scores.masked_fill(mask, float("-inf"))

        ## softmax + dropout
        #attn_weights = F.softmax(scores, dim=-1)
        #attn_weights_dropped = self.dropout(attn_weights)

        ## heads values -> (batch, n_heads, seq_q, d_head)
        #context = attn_weights_dropped @ V
        #context = self._reshape_from_heads(context)
        #output = self.W_O(context)

        #if return_weights: return output, attn_weights
        #return output

        ##########################################################
        #################  USE  FLASH ATTENTION  #################
        ##########################################################
        if return_weights:
            scores = (Q @ K.transpose(-2, -1)) / self.scale
            if causal:
                causal_mask = torch.triu(torch.ones(seq_q, seq_k, device=scores.device, dtype=torch.bool), diagonal=1)
                scores = scores.masked_fill(causal_mask, float("-inf"))
            if mask is not None:
                scores = scores.masked_fill(mask, float("-inf"))
            attn_weights = F.softmax(scores, dim=-1)
            context = self.dropout(attn_weights) @ V
            output = self.W_O(self._reshape_from_heads(context))
            return output, attn_weights

        # Flash Attention: O(N) memory, 2-4× faster
        context = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=None if causal else mask,
            is_causal=causal,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        output = self.W_O(self._reshape_from_heads(context))
        return output


    def _reshape_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq, d_model) -> (batch, n_heads, seq, d_head)"""
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.n_heads, self.d_head)
        return x.transpose(1, 2)

    def _reshape_from_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, n_heads, seq, d_head) -> (batch, seq, d_model)"""
        batch, _, seq, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq, self.d_model)
