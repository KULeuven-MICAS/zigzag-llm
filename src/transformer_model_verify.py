# Adapted from https://github.com/suvash/nnze2he/blob/main/makemore/src/gpt.py
# Verify stage for speculative decoding
# [TODO] Chao: Double check if the verify stage works correctly.


import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from src.config import LLMConfig

device = "cpu"
dropout = 0.3


class Matmul(nn.Module):
    """Wrap Torch Matmul operator so that the operation can be given a custom name that is exported to ONNX"""

    def forward(self, a: Tensor, b: Tensor):
        return a @ b


class Head(nn.Module):
    """one head of self attention"""

    def __init__(self, cfg: LLMConfig):
        super().__init__()  # type: ignore
        self.cfg = cfg

        self.mul_qk_t = Matmul()
        self.mul_logits_v = Matmul()

        # Causal mask for verify_size tokens attending to (prefill_size + verify_size) tokens
        total_size = cfg.prefill_size + cfg.verify_size
        self.register_buffer("tril", torch.tril(torch.ones(total_size, total_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, key_token: Tensor, query_token: Tensor, value_token: Tensor):
        _, L, _ = key_token.shape  # (B, verify_size, head_size)

        # Load KV cache from prefill stage
        key_cache = torch.ones((self.cfg.batch_size, self.cfg.prefill_size, self.cfg.head_size))
        full_key = torch.cat((key_cache, key_token), dim=1)  # (B, prefill_size+verify_size, d_h)
        key_transpose = full_key.transpose(-2, -1)

        value_cache = torch.ones((self.cfg.batch_size, self.cfg.prefill_size, self.cfg.head_size))
        full_value = torch.cat((value_cache, value_token), dim=1)  # (B, prefill_size+verify_size, d_h)

        # Attention matrix for verify_size queries attending to all keys
        # (B, verify_size, d_h) @ (B, d_h, prefill_size+verify_size) -> (B, verify_size, prefill_size+verify_size)
        attention: Tensor = self.mul_qk_t(query_token, key_transpose)
        attention = attention / math.sqrt(self.cfg.head_size)
        
        # Apply causal masking: only mask within the verify region
        # The first verify_size rows start from prefill_size position
        mask = self.tril[self.cfg.prefill_size : self.cfg.prefill_size + L, : self.cfg.prefill_size + L]
        attention = attention.masked_fill(mask == 0, float("-inf"))  # (B, verify_size, prefill_size+verify_size)

        logits = F.softmax(attention, dim=-1)  # (B, verify_size, prefill_size+verify_size)
        logits = self.dropout(logits)
        out = self.mul_logits_v(logits, full_value)  # (B, verify_size, d_h)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: LLMConfig):
        super().__init__()  # type: ignore
        self.cfg = cfg
        self.heads = nn.ModuleList([Head(cfg) for _ in range(cfg.num_head)])

        # We compute each linear projection as one big MatMul to ensure spatial array utilization
        self.key_proj = nn.Linear(cfg.embedding_dim, cfg.embedding_dim, bias=False)
        self.query_proj = nn.Linear(cfg.embedding_dim, cfg.embedding_dim, bias=False)
        self.value_proj = nn.Linear(cfg.embedding_dim, cfg.embedding_dim, bias=False)

        # NOTE  `num_head * head_size` must equal `embedding_dim`
        self.out_proj = nn.Linear(cfg.num_head * cfg.head_size, cfg.embedding_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        # `cfg.num_head` might be changed to shorten simulation time -> recompute the correct dimension
        num_head_tensors = self.cfg.embedding_dim // self.cfg.head_size

        # (B, verify_size, num_head_tensors, d_h)
        key: Tensor = self.key_proj(x).reshape(
            self.cfg.batch_size, self.cfg.verify_size, num_head_tensors, self.cfg.head_size
        )
        query: Tensor = self.query_proj(x).reshape(
            self.cfg.batch_size, self.cfg.verify_size, num_head_tensors, self.cfg.head_size
        )
        value: Tensor = self.value_proj(x).reshape(
            self.cfg.batch_size, self.cfg.verify_size, num_head_tensors, self.cfg.head_size
        )

        out = torch.cat(
            [head(key[:, :, idx, :], query[:, :, idx, :], value[:, :, idx, :]) for idx, head in enumerate(self.heads)],
            dim=-1,
        )
        out = self.out_proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    "simple linear layer followed by non linearity"

    def __init__(self, cfg: LLMConfig):
        super().__init__()  # type: ignore
        self.feedforward_expand = nn.Linear(cfg.embedding_dim, cfg.dim_ff, bias=True)
        self.feedforward_contract = nn.Linear(cfg.dim_ff, cfg.embedding_dim, bias=True)

        self.net = nn.Sequential(
            self.feedforward_expand,
            nn.ReLU(),
            self.feedforward_contract,
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor):
        return self.net(x)


class Block(nn.Module):
    """a transformer block : communication then computation"""

    def __init__(self, cfg: LLMConfig):
        super().__init__()  # type: ignore

        self.sa = MultiHeadAttention(cfg)
        self.feed_forward = FeedForward(cfg)
        self.layer_norm1 = nn.LayerNorm(cfg.embedding_dim)
        self.layer_norm2 = nn.LayerNorm(cfg.embedding_dim)

    def forward(self, x: Tensor):
        x = x + self.sa(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class LanguageModelVerify(nn.Module):
    """Run inference in the verify stage for speculative decoding. Verifies verify_size tokens 
    generated by the draft model, with access to KV cache from prefill stage."""

    def __init__(self, cfg: LLMConfig):
        super().__init__()  # type: ignore
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        # Use relative position embeddings (same pattern as decode and prefill)
        self.position_embedding_table = nn.Embedding(cfg.verify_size, cfg.embedding_dim)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.num_layer)])
        self.layer_norm_final = nn.LayerNorm(cfg.embedding_dim)
        self.de_embed = nn.Linear(cfg.embedding_dim, cfg.vocab_size)

    def forward(self, idx: Tensor):
        _, L = idx.shape  # Should be (B, verify_size)

        # idx is (B, verify_size) tensor of integers representing draft tokens to verify
        token_emb = self.token_embedding_table(idx)  # (B, verify_size, d)
        
        # Position embeddings use relative positions [0, verify_size-1]
        pos_emb = self.position_embedding_table(torch.arange(L, device=device))  # (verify_size, d)
        
        x = token_emb + pos_emb  # (B, verify_size, d)
        x = self.blocks(x)
        x = self.layer_norm_final(x)
        logits = self.de_embed(x)  # (B, verify_size, VOCAB_SIZE)

        return logits