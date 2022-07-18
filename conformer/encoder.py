# Reproducing the model from the original Conformer paper
# First attempt may be rough
# Default values based on Conformer L

import torch.nn as nn
from subsampling import Conv2dSubsampling
from conformer_block import ConformerBlock
from typing import Tensor, Tuple

class Conformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 512,
        num_layers: int = 17,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True
    ) -> None:
        super(Conformer, self).__init__()

        self.subsampling = Conv2dSubsampling(in_channels=1, out_channels=encoder_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(in_features=(((input_dim - 1) // 2 - 1) // 2), out_features=encoder_dim),
            nn.Dropout(p=input_dropout_p)
        )
        self.conformer_blocks = nn.ModuleList([ConformerBlock(
            encoder_dim=encoder_dim,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual
        ) for _ in range(num_layers)])

    def count_parameters(self) -> int:
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        outputs, output_lengths = self.subsampling(inputs=inputs, input_lengths=input_lengths)
        outputs = self.input_proj(outputs)

        for conformer_block in self.conformer_blocks:
            outputs = conformer_block(outputs)

        return outputs, output_lengths