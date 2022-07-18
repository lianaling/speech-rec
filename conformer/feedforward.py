# %% [markdown]

# # Feedforward module

# [The expansion ratio, defined as the number of hidden units over the number of input units](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiVls-vp-j4AhUe-jgGHbfrBqAQFnoECCkQAQ&url=https%3A%2F%2Fui.adsabs.harvard.edu%2Fabs%2FarXiv%3A1705.07441&usg=AOvVaw3-SlI1G7AxoBYz2sG2DxuX)

import torch.nn as nn
from typing import Tensor

class FeedForwardModule(nn.Module):
    def __init__(self, encoder_dim: int, expansion_factor: int, dropout_p: float = 0.2) -> None:
        super(FeedForwardModule, self).__init__()

        self.sequential = nn.Sequential(
            nn.LayerNorm(),
            # The first linear layer uses an expansion factor of 4
            nn.Linear(in_features=encoder_dim, out_features=encoder_dim * expansion_factor),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            # The second linear layer projects it back to the model dimension
            nn.Linear(in_features=encoder_dim * expansion_factor, out_features=encoder_dim),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)