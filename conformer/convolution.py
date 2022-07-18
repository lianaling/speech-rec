import torch.nn as nn
from typing import Tensor

from modules import Transpose

class ConvolutionModule(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 31, expansion_factor: int = 2, dropout_p: float = 0.2) -> None:
        super(ConvolutionModule, self).__init__()

        assert (kernel_size - 1) % 2 == 0, 'kernel_size should be an odd number for SAME padding'
        assert expansion_factor == 2, 'Currently only supports expansion factor 2'

        self.sequential = nn.Sequential(
            nn.LayerNorm(normalized_shape=in_channels),
            Transpose(shape=(1, 2)),
            PointwiseConv1d(in_channels=in_channels, out_channels=in_channels * expansion_factor, stride=1, padding=0, bias=True),
            nn.GLU(dim=1),
            DepthwiseConv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(num_features=in_channels),
            nn.SiLU(),
            PointwiseConv1d(in_channels=in_channels, out_channels=in_channels, stride=1, padding=0, bias=True),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs).transpose(1, 2)

class DepthwiseConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False) -> None:
        super(DepthwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

# kernel_size = 1 means pointwise conv
class PointwiseConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, padding: int = 0, bias: bool = True) -> None:
        super(PointwiseConv1d, self).__init__()

        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding, bias=bias),

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)