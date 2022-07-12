# Reproducing the model from the original Conformer paper
# First attempt may be rough
# Default values based on Conformer L

from torch import nn
from subsampling import Conv2dSubsampling
from conformer_block import ConformerBlock

class Conformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 512,
        num_layers: int = 17,
        num_attention_heads: int = 8,
        conv_kernel_size: int = 32,
        p_dropout: float = 0.1
    ) -> None:
        super(Conformer, self).__init__()

        self.subsampling = Conv2dSubsampling(in_channels=1)
        self.sequential = nn.Sequential(
            nn.Linear(),
            nn.Dropout(p=p_dropout)
        )

        self.conformer_blocks = nn.ModuleList([ConformerBlock() for i in range(num_layers)])
