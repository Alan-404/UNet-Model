import torch
from torch import Tensor
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv2d_1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1, bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(num_features=mid_channels)

        self.activation = nn.ReLU(inplace=True)

        self.conv2d_2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        self.batch_norm_2 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x: Tensor):
        x = self.conv2d_1(x)
        x = self.batch_norm_1(x)
        x = self.activation(x)
        x = self.conv2d_2(x)
        x = self.batch_norm_2(x)
        x = self.activation(x)
        return x
    
class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
    def forward(self, x: Tensor):
        return self.conv(x)