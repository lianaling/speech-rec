# %% [markdown]
# [Transformers Explained Visually (Part 3): Multi-head Attention, deep dive](https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853)

from email.headerregistry import DateHeader
from numpy import transpose
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from typing import Tensor

class RelativeMultiHeadAttention(nn.Module):
    # d_model is model dimension; embedding size of the input data
    # query size is usually equal to key and value size
    # num of attention heads
    # d_model % num_heads == 0
    # d_head is dimensions per head
    # sqrt_dim is the div term in the attention formula softmax(QK^t/sqrt(d_model))V
    # batch_size gives one dimension for number of samples
    def __init__(self, d_model: int = 512, num_heads: int = 16, dropout_p: float = 0.1) -> None:
        super(RelativeMultiHeadAttention, self).__init__()

        assert(d_model % num_heads == 0, 'd_model % num_heads should be zero (divisible)')
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        # Projections
        self.query_proj = nn.Linear(in_features=d_model, out_features=d_model)
        self.key_proj = nn.Linear(in_features=d_model, out_features=d_model)
        self.value_proj = nn.Linear(in_features=d_model, out_features=d_model)
        self.pos_proj = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        # Bias terms
        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, pos_embedding: Tensor, mask: Tensor = None) -> Tensor:
        batch_size = value.size()

        # Reorder dimensions for k and v: batch_size, num_heads, -1, d_heads
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        # Q * K^t (transpose)
        # batch_size, num_heads, -1, d_head matmul batch_size, d_heads, -1, num_heads
        # num_heads * d_heads = d_model
        # -1 * -1 will give the score
        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        # Add relative positional embedding to query (inputs)
        # Reorder dimensions for pos_embedding: batch_size, d_head, -1, num_heads
        # batch_size, num_heads, -1, d_head matmul batch_size, d_head, -1, num_heads
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)

        # QK^t/sqrt(d_model)
        # Add score from content and position to give positional information
        score = (content_score + pos_score) / self.sqrt_dim

        # Mask attention scores for auto-regressive property
        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)

        # softmax(QK^t/sqrt(d_model))
        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        # softmax(QK^t/sqrt(d_model))V
        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score