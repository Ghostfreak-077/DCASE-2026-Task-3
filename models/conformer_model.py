import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# Feed Forward Module
# --------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------
# Convolution Module
# --------------------------------------------------
class ConformerConvModule(nn.Module):
    def __init__(self, dim, kernel_size=31, dropout=0.1):
        super().__init__()

        self.ln = nn.LayerNorm(dim)

        self.pw1 = nn.Conv1d(dim, dim * 2, kernel_size=1)

        self.dw = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim
        )

        self.bn = nn.BatchNorm1d(dim)

        self.pw2 = nn.Conv1d(dim, dim, kernel_size=1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B,N,C)

        x = self.ln(x)

        x = x.transpose(1, 2)

        x = self.pw1(x)

        x, gate = x.chunk(2, dim=1)
        x = x * torch.sigmoid(gate)

        x = self.dw(x)
        x = self.bn(x)
        x = F.silu(x)

        x = self.pw2(x)
        x = self.dropout(x)

        return x.transpose(1, 2)


# --------------------------------------------------
# Single Conformer Block
# --------------------------------------------------
class ConformerBlock(nn.Module):
    def __init__(
        self,
        dim=512,
        heads=8,
        ff_mult=4,
        dropout=0.1,
        conv_kernel=31,
    ):
        super().__init__()

        self.ff1 = FeedForward(dim, ff_mult, dropout)

        self.attn_ln = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

        self.conv = ConformerConvModule(
            dim,
            kernel_size=conv_kernel,
            dropout=dropout,
        )

        self.ff2 = FeedForward(dim, ff_mult, dropout)

        self.final_ln = nn.LayerNorm(dim)

    def forward(self, x):

        x = x + 0.5 * self.ff1(x)

        attn_in = self.attn_ln(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)

        x = x + attn_out

        x = x + self.conv(x)

        x = x + 0.5 * self.ff2(x)

        return self.final_ln(x)


# --------------------------------------------------
# Encoder
# --------------------------------------------------
class ConformerEncoder(nn.Module):
    def __init__(
        self,
        dim=512,
        depth=4,
        heads=8,
        ff_mult=4,
        dropout=0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            ConformerBlock(
                dim=dim,
                heads=heads,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        # x : (B,N,C)

        for layer in self.layers:
            x = layer(x)

        return x