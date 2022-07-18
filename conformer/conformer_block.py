from turtle import forward
from torch import HalfStorage
import torch.nn as nn
from feedforward import FeedForwardModule
from attention import MultiHeadedSelfAttentionModule
from convolution import ConvolutionModule
from residual_connection import ResidualConnection
from typing import Tensor

class ConformerBlock(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 512,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True
    ) -> None:
        super(ConformerBlock, self).__init__()

        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        self.sequential = nn.Sequential(
            FeedForwardModule(encoder_dim=encoder_dim, expansion_factor=feed_forward_expansion_factor, dropout_p=feed_forward_dropout_p),
            ResidualConnection(
                MultiHeadedSelfAttentionModule(d_model=encoder_dim, num_heads=num_attention_heads, dropout_p=attention_dropout_p),
                module_factor=self.feed_forward_residual_factor
                ),
            ResidualConnection(
                ConvolutionModule(in_channels=encoder_dim, kernel_size=conv_kernel_size, expansion_factor=conv_expansion_factor, dropout_p=conv_dropout_p)
                ),
            ResidualConnection(
                FeedForwardModule(encoder_dim=encoder_dim, expansion_factor=feed_forward_expansion_factor, dropout_p=feed_forward_dropout_p),
                module_factor=self.feed_forward_residual_factor
                )
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)