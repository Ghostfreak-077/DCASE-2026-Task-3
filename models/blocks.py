import torch
import torch.nn as nn
import torch.nn.functional as F

# ════════════════════════════════════════════════════════════════════════════
# 1.  BUILDING BLOCKS
# ════════════════════════════════════════════════════════════════════════════

def _circ_pad2d(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Circular pad on W (azimuth), reflect on H (elevation)."""
    x = F.pad(x, (pad, pad, 0,   0  ), mode="circular")
    x = F.pad(x, (0,   0,   pad, pad), mode="reflect")
    return x


class _CircConv2d(nn.Module):
    """Conv2d with circular-W + reflect-H padding; supports asymmetric stride."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=(1, 1), bias=False):
        super().__init__()
        if isinstance(stride, int):
            stride = (stride, stride)
        self.pad    = kernel // 2
        self.stride = stride
        self.conv   = nn.Conv2d(in_ch, out_ch, kernel,
                                stride=stride, padding=0, bias=bias)

    def forward(self, x):
        if self.pad > 0:
            x = _circ_pad2d(x, self.pad)
        return self.conv(x)


class _ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, stride=(1, 1)):
        super().__init__()
        self.block = nn.Sequential(
            _CircConv2d(in_ch, out_ch, stride=stride),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _EncoderBlock(nn.Module):
    """
    Two conv-bn-relu layers.
    First conv applies the given stride; second is always stride (1,1).
    stride=(1,2) downsamples F (width) only — preserving T (height).
    """
    def __init__(self, in_ch, out_ch, stride=(1, 1), dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            _ConvBnRelu(in_ch,  out_ch, stride=stride),
            _ConvBnRelu(out_ch, out_ch, stride=(1, 1)),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class _SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, ch, reduction=8):
        super().__init__()
        mid = max(1, ch // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x).view(x.shape[0], -1, 1, 1)


class _DecoderBlock(nn.Module):
    """Upsample → concat SE-gated skip → two conv-bn-relu."""
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.1):
        super().__init__()
        self.se   = _SEBlock(skip_ch)
        self.conv = nn.Sequential(
            _ConvBnRelu(in_ch + skip_ch, out_ch),
            _ConvBnRelu(out_ch,          out_ch),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x, skip, target_size):
        x    = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        skip = self.se(skip)
        return self.conv(torch.cat([x, skip], dim=1))

