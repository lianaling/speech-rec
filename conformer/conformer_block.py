from re import finditer
from torch import nn
from feedforward import FeedForwardModule
from attention import AttentionModule
from convolution import ConvolutionModule

class ConformerBlock(nn.Module):
    def __init__(self) -> None:
        super(ConformerBlock, self).__init__()

        self.ff = FeedForwardModule()
        self.attn = ResidualConnection(AttentionModule())
        self.conv = ResidualConnection(ConvolutionModule())
        self.out = ResidualConnection(FeedForwardModule())