import torch.nn as nn
from models.blocks import (
    _CircConv2d,
    _ConvBnRelu
)
import torch.nn.functional as F

class _EnergyMapHead(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.head = nn.Sequential(
            _ConvBnRelu(in_ch, in_ch),
            _CircConv2d(in_ch, n_classes, kernel=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.head(x)


class _InstanceMaskHead(nn.Module):
    def __init__(self, in_ch, n_classes, dropout=0.2):
        super().__init__()
        self.head = nn.Sequential(
            _ConvBnRelu(in_ch, in_ch),
            nn.Dropout2d(dropout),
            _CircConv2d(in_ch, n_classes, kernel=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.head(x)


class _DistanceHead(nn.Module):
    """Global avg-pool of bottleneck → MLP → (B, N_CLASSES)."""
    def __init__(self, bottleneck_ch, n_classes, hidden=256, dropout=0.3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp  = nn.Sequential(
            nn.Linear(bottleneck_ch, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, bottleneck):
        return F.softplus(self.mlp(self.pool(bottleneck).flatten(1)))
