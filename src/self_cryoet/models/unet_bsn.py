import torch
import torch.nn as nn

from .modules.blocks import DecoderStage, EncoderStage
from .modules.masked_conv3d import SparseMaskedConv3d
from .modules.volume_shuffle import VolumeShuffle, VolumeUnshuffle


class UShapeBSN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 24,
        num_levels: int = 2,
        blocks_per_level: int = 2,
        shuffle_factor: int = 3,
    ):
        super().__init__()
        if num_levels != 2:
            raise ValueError("This initial implementation fixes the network to 2 levels for clarity.")

        self.stem = nn.Sequential(
            SparseMaskedConv3d(in_channels, base_channels, kernel_size=5),
            nn.GELU(),
        )

        self.enc1 = EncoderStage(base_channels, num_blocks=blocks_per_level)
        self.down1 = VolumeUnshuffle(shuffle_factor)
        self.down1_proj = nn.Conv3d(base_channels * shuffle_factor**3, base_channels * 2, kernel_size=1)

        self.enc2 = EncoderStage(base_channels * 2, num_blocks=blocks_per_level)
        self.down2 = VolumeUnshuffle(shuffle_factor)
        self.down2_proj = nn.Conv3d(base_channels * 2 * shuffle_factor**3, base_channels * 4, kernel_size=1)

        self.bottleneck = EncoderStage(base_channels * 4, num_blocks=blocks_per_level)

        self.up2_proj = nn.Conv3d(base_channels * 4, base_channels * 2 * shuffle_factor**3, kernel_size=1)
        self.up2 = VolumeShuffle(shuffle_factor)
        self.dec2 = DecoderStage(base_channels * 4, num_blocks=blocks_per_level)
        self.dec2_proj = nn.Conv3d(base_channels * 4, base_channels * 2, kernel_size=1)

        self.up1_proj = nn.Conv3d(base_channels * 2, base_channels * shuffle_factor**3, kernel_size=1)
        self.up1 = VolumeShuffle(shuffle_factor)
        self.dec1 = DecoderStage(base_channels * 2, num_blocks=blocks_per_level)
        self.dec1_proj = nn.Conv3d(base_channels * 2, base_channels, kernel_size=1)

        self.head = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(base_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        e1 = self.enc1(x0)

        x1 = self.down1(e1)
        x1 = self.down1_proj(x1)
        e2 = self.enc2(x1)

        x2 = self.down2(e2)
        x2 = self.down2_proj(x2)
        b = self.bottleneck(x2)

        d2 = self.up2_proj(b)
        d2 = self.up2(d2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d2 = self.dec2_proj(d2)

        d1 = self.up1_proj(d2)
        d1 = self.up1(d1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        d1 = self.dec1_proj(d1)

        return self.head(d1)

