from torch import nn
from typing import Tensor

from modules import Transpose

class ConvolutionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 31, drop_p: float = 0.2) -> None:
        super(ConvolutionModule, self).__init__()

        self.expansion_factor = 2

        self.sequential = nn.Sequential(
            nn.LayerNorm(normalized_shape=in_channels),
            # Transpose(shape=(1,2)),
            # TODO: Add in tranpose
            PointwiseConv1d(in_channels=in_channels, out_channels=in_channels * self.expansion_factor),
            nn.GLU(), # TODO: change dim=1 instead of default dim=-1
            DepthwiseConv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size),
            nn.BatchNorm1d(num_features=in_channels),
            nn.SiLU(),
            PointwiseConv1d(),
            nn.Dropout(p=drop_p)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs) # TODO: add in transpose

class DepthwiseConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super(DepthwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class PointwiseConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(PointwiseConv1d, self).__init__()

        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)