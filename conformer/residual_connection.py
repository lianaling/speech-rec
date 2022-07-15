import torch.nn as nn
from typing import Tensor

class ResidualConnection(nn.Module):
    '''outputs = (module(inputs) * module_factor + inputs * input_factor)'''

    def __init__(self, module: nn.Module, module_factor: float = 1.0, input_factor: float = 1.0) -> None:
        super(ResidualConnection, self).__init__()

        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs: Tensor) -> Tensor:
        return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)