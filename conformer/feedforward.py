# %% [markdown]

# # Feedforward module

# [The expansion ratio, defined as the number of hidden units over the number of input units](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiVls-vp-j4AhUe-jgGHbfrBqAQFnoECCkQAQ&url=https%3A%2F%2Fui.adsabs.harvard.edu%2Fabs%2FarXiv%3A1705.07441&usg=AOvVaw3-SlI1G7AxoBYz2sG2DxuX)

# %%
from torch import nn
from typing import Tensor

class FeedForwardModule(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, drop_p: float = 0.2) -> None:
        super(FeedForwardModule, self).__init__()

        hidden_dim = input_dim * 4 # Expansion factor of 4 = hidden_dim * 4?

        self.sequential = nn.Sequential(
            nn.LayerNorm(),
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=drop_p),
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
            nn.Dropout(p=drop_p)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)