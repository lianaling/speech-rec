# %% [markdown]

# # Multi-headed self-attention module
# [MHA with relative positional encoding](https://github.com/kimiyoung/transformer-xl/blob/44781ed21dbaec88b280f74d9ae2877f52b492a5/pytorch/mem_transformer.py#L293)

# %%
from torch import nn
from typing import Tensor

class AttentionModule(nn.Module):
    def __init__(self, drop_p: float = 0.2) -> None:
        super(AttentionModule, self).__init__()

        self.sequential = nn.Sequential(
            nn.LayerNorm(),
            nn.MultiheadAttention(), # TODO: change to MHA with relative positional encodings
            nn.Dropout(p=drop_p)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)