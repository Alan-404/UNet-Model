import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from .conv import DoubleConv

class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.max_conv2d = nn.MaxPool2d(kernel_size=2)
        self.double_conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x: Tensor):
        x = self.max_conv2d(x)
        x = self.double_conv(x)
        return x
    
class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels, mid_channels=in_channels//2)
        else:
            self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x1: Tensor, x2: Tensor):
        x1 = self.up(x1)

        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)