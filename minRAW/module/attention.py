import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch.nn import functional as F
import math

Past = Tuple[torch.Tensor, torch.Tensor]


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, heads, dims, seq_len, dropout):
        super().__init__()
        # print(dims,heads)
        assert dims % heads == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(dims, dims)
        self.query = nn.Linear(dims, dims)
        self.value = nn.Linear(dims, dims)
        self.proj = nn.Linear(dims, dims)
        # regularization
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        # output projection

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.n_head = heads

    def forward(self, x, mask):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # print(x.dtype, self.key.weight.dtype)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = att.masked_fill(mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
