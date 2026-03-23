import torch.nn as nn

from .dca_block import DCABlock


class EncoderStage(nn.Module):
    def __init__(self, channels: int, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.Sequential(*[DCABlock(channels) for _ in range(num_blocks)])

    def forward(self, x):
        return self.blocks(x)


class DecoderStage(nn.Module):
    def __init__(self, channels: int, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.Sequential(*[DCABlock(channels) for _ in range(num_blocks)])

    def forward(self, x):
        return self.blocks(x)

