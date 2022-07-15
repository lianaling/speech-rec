# %% [markdown]

# # Multi-headed self-attention module
# [MHA with relative positional encoding](https://github.com/kimiyoung/transformer-xl/blob/44781ed21dbaec88b280f74d9ae2877f52b492a5/pytorch/mem_transformer.py#L293)

import torch.nn as nn
from typing import Tensor
from pos_encoding import PositionalEncoding
from rmh_attention import RelativeMultiHeadAttention

class MultiHeadedSelfAttentionModule(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1) -> None:
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model=d_model)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
        self.attn = RelativeMultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout_p=dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Tensor = None) -> Tensor:
        batch_size, seq_length, _ = inputs.size() # Where batch first = True
        pos_embedding = self.positional_encoding(length=seq_length)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        inputs = self.layer_norm(inputs)
        outputs = self.attn(query=inputs, key=inputs, value=inputs, pos_embedding=pos_embedding, mask=mask)

        return self.dropout(outputs)