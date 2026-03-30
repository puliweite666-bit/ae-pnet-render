import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=4)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            diff = skip.shape[-1] - x.shape[-1]
            if diff > 0:
                x = F.pad(x, (0, diff))
            elif diff < 0:
                x = x[..., :skip.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class AEPNet(nn.Module):
    """
    PhaseNet 风格 1D U-Net
    输入: [B, 3, T]
    输出: [B, 2, T] -> [noise_prob, p_prob]
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 16, num_classes: int = 2):
        super().__init__()
        self.d1 = DownBlock(in_channels, base_channels)
        self.d2 = DownBlock(base_channels, base_channels * 2)
        self.d3 = DownBlock(base_channels * 2, base_channels * 4)
        self.d4 = DownBlock(base_channels * 4, base_channels * 8)

        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)

        self.u4 = UpBlock(base_channels * 16, base_channels * 8, base_channels * 8)
        self.u3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.u2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.u1 = UpBlock(base_channels * 2, base_channels, base_channels)

        self.head = nn.Conv1d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        s1, x = self.d1(x)
        s2, x = self.d2(x)
        s3, x = self.d3(x)
        s4, x = self.d4(x)
        x = self.bottleneck(x)
        x = self.u4(x, s4)
        x = self.u3(x, s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)
        logits = self.head(x)
        probs = torch.softmax(logits, dim=1)
        return logits, probs
