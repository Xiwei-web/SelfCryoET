import torch
import torch.nn as nn

from .channel_attention import SimpleChannelAttention
from .simple_gate import SimpleGate


class DilatedDepthwiseConv3d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 3):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DCABlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2, dilation: int = 3):
        super().__init__()
        hidden = channels * expansion
        self.norm = nn.GroupNorm(1, channels)
        self.pre = nn.Conv3d(channels, hidden * 2, kernel_size=1)
        self.ddc = DilatedDepthwiseConv3d(hidden * 2, dilation=dilation)
        self.gate = SimpleGate()
        self.sca = SimpleChannelAttention(hidden)
        self.post = nn.Conv3d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.pre(x)
        x = self.ddc(x)
        x = self.gate(x)
        x = self.sca(x)
        x = self.post(x)
        return x + residual

