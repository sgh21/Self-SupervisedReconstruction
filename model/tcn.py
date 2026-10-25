"""TCN autoencoder for three-channel vibration velocity waveforms."""

import torch
from torch import nn
from torch.nn import functional as F

from model.tcn_standard import StandardTCNAutoencoder


class TCNBlock(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, dilation: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.input = nn.Conv1d(channels, hidden_channels * 2, 1)
        self.depthwise = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            3,
            padding=dilation,
            dilation=dilation,
            groups=hidden_channels,
        )
        self.output = nn.Conv1d(hidden_channels, channels, 1)
        self.norm = nn.GroupNorm(1, hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        value, gate = self.input(signal).chunk(2, dim=1)
        hidden = self.depthwise(value) * torch.sigmoid(gate)
        return signal + self.output(self.dropout(F.gelu(self.norm(hidden))))


class TCNAutoencoder(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        hidden_channels: int = 76,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 1, 2),
        kernel_size: int = 64,
        stride: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if stride < 1 or kernel_size < stride or (kernel_size - stride) % 2:
            raise ValueError(
                "stride must be positive; kernel_size must be >= stride "
                "with an even difference"
            )
        padding = (kernel_size - stride) // 2
        self.stride = stride
        self.encoder = nn.Sequential(
            nn.Conv1d(3, channels, kernel_size, stride=stride, padding=padding),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(
            *(TCNBlock(channels, hidden_channels, dilation, dropout) for dilation in dilations)
        )
        self.decoder = nn.ConvTranspose1d(
            channels, 3, kernel_size, stride=stride, padding=padding
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 3 or signal.shape[1] != 3:
            raise ValueError("input must have shape [batch, 3, time]")
        if signal.shape[-1] % self.stride:
            raise ValueError(f"input length must be divisible by {self.stride}")
        return self.decoder(self.temporal(self.encoder(signal)))


def build_autoencoder(
    implementation: str = "custom",
    channels: int = 64,
    hidden_channels: int = 76,
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 1, 2),
    kernel_size: int = 64,
    stride: int = 32,
    standard: dict | None = None,
    dropout: float = 0.0,
) -> nn.Module:
    """Use one encoder/decoder width and the same block count for both models."""
    if implementation == "custom":
        return TCNAutoencoder(channels, hidden_channels, dilations, kernel_size, stride, dropout)
    if implementation == "standard":
        return StandardTCNAutoencoder(
            encoder_channels=channels,
            encoder_kernel_size=kernel_size,
            stride=stride,
            num_channels=[channels] * len(dilations),
            **(standard or {}),
        )
    raise ValueError(f"Unknown TCN implementation: {implementation!r}; use custom or standard")
