import torch
import torch.nn as nn
from seg.registry.registry import MODELS


@MODELS.register_module()
class EX_KD(nn.Module):
    def __init__(self,
                 in_channels: int,
                 ratio: int = 4):
        super().__init__()

        channels = in_channels // ratio

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=channels,
                kernel_size=(1, 1)),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(1, 3),
                padding=(0, 1)),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(3, 1),
                padding=(1, 0)),
            nn.Conv2d(
                in_channels=channels,
                out_channels=in_channels,
                kernel_size=(1, 1)))

        self.conv2 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        attention = self.conv1(x)
        attention = self.conv2(attention)
        return attention * x
