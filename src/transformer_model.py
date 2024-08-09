# Adapted from https://github.com/suvash/nnze2he/blob/main/makemore/src/gpt.py

import math
import torch
from torch import Tensor
import torch.nn as nn
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

        # in Pytorch convention a variable that's not a parameter of the model is called a buffer
        self.register_buffer("tril", torch.tril(torch.ones(cfg.prefill_size, cfg.prefill_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, key: Tensor, query: Tensor, value: Tensor):

        _, L, _ = key.shape
        key_transpose = key.transpose(-2, -1)

        attention: Tensor = self.mul_qk_t(query, key_transpose)  # (B, L, d_h) @ (B, d_h, L) -> (B, L, L)
        attention = attention / math.sqrt(self.cfg.head_size)
        attention = attention.masked_fill(self.tril[:L, :L] == 0, float("-inf"))  # (B, L, L)

        logits = F.softmax(attention, dim=-1)  # (B, L, L)
        logits = self.dropout(logits)
        out = self.mul_logits_v(logits, value)
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

        # (B, L, num_head_tensors, d_h)
        key: Tensor = self.key_proj(x).reshape(
            self.cfg.batch_size, self.cfg.prefill_size, num_head_tensors, self.cfg.head_size
        )
        query: Tensor = self.query_proj(x).reshape(
            self.cfg.batch_size, self.cfg.prefill_size, num_head_tensors, self.cfg.head_size
        )
        value: Tensor = self.value_proj(x).reshape(
            self.cfg.batch_size, self.cfg.prefill_size, num_head_tensors, self.cfg.head_size
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


class LanguageModel(nn.Module):

    def __init__(self, cfg: LLMConfig):
        super().__init__()  # type: ignore
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        self.position_embedding_table = nn.Embedding(cfg.prefill_size, cfg.embedding_dim)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.num_layer)])
        self.layer_norm_final = nn.LayerNorm(cfg.embedding_dim)
        self.de_embed = nn.Linear(cfg.embedding_dim, cfg.vocab_size)

    def forward(self, idx: Tensor):
        _, L = idx.shape

        # idx and targets are both (B, L) tensor of integers
        token_emb = self.token_embedding_table(idx)  # (B, L, C)
        pos_emb = self.position_embedding_table(torch.arange(L, device=device))  # (L, C)
        x = token_emb + pos_emb  # (B, L, C)
        x = self.blocks(x)
        x = self.layer_norm_final(x)
        logits = self.de_embed(x)  # (B,L,VOCAB_SIZE)

        return logits
