import torch.nn as nn
from typing import Tensor

class Transpose(nn.Module):
    def __init__(self, shape: tuple) -> None:
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)