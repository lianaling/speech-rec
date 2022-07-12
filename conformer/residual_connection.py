from torch import nn
from typing import Tensor

class ResidualConnection(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super(ResidualConnection, self).__init__()

        self.module = module

    def forward(self, residual: Tensor) -> Tensor:
        return self.module