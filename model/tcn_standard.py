"""Independent standard-TCN autoencoder; depends only on PyTorch.

The causal residual blocks follow Bai, Kolter and Koltun (2018),
https://arxiv.org/abs/1803.01271 and https://github.com/locuslab/TCN.
The strided encoder and decoder are this project's reconstruction adaptation.
"""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class Chomp1d(nn.Module):
    """Remove right padding so each output uses only current and past inputs."""

    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return signal[..., :-self.size].contiguous() if self.size else signal


class TemporalBlock(nn.Module):
    """Two weight-normalized causal convolutions and a residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
        spatial_dropout: bool = True,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        )
        conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        )
        # Initialize the underlying weights before applying WeightNorm.
        nn.init.normal_(conv1.weight, mean=0.0, std=0.01)
        nn.init.normal_(conv2.weight, mean=0.0, std=0.01)
        dropout_layer = nn.Dropout1d if spatial_dropout else nn.Dropout
        self.net = nn.Sequential(
            weight_norm(conv1), Chomp1d(padding), nn.ReLU(), dropout_layer(dropout),
            weight_norm(conv2), Chomp1d(padding), nn.ReLU(), dropout_layer(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, mean=0.0, std=0.01)
        self.relu = nn.ReLU()

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        residual = signal if self.downsample is None else self.downsample(signal)
        return self.relu(self.net(signal) + residual)


class TemporalConvNet(nn.Module):
    """Stack standard TCN blocks with dilation 1, 2, 4, ... ."""

    def __init__(
        self,
        num_inputs: int,
        num_channels: Sequence[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
        spatial_dropout: bool = True,
    ) -> None:
        super().__init__()
        self.num_channels = tuple(num_channels)
        if not self.num_channels or min(self.num_channels) < 1 or kernel_size < 1:
            raise ValueError("num_channels must contain positive widths; kernel_size must be positive")
        layers = []
        for index, out_channels in enumerate(self.num_channels):
            in_channels = num_inputs if index == 0 else self.num_channels[index - 1]
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, 2**index,
                dropout, spatial_dropout,
            ))
        self.network = nn.Sequential(*layers)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.network(signal)


class StandardTCNAutoencoder(nn.Module):
    """[B, 3, T] -> compressed features -> [B, 3, T].

    The standard TCN backbone is causal. The complete autoencoder, including
    its strided encoder and transposed-convolution decoder, is not causal.
    """

    def __init__(
        self,
        encoder_channels: int = 64,
        encoder_kernel_size: int = 64,
        stride: int = 32,
        num_channels: Sequence[int] = (64, 64, 64, 64, 64, 64, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.2,
        spatial_dropout: bool = True,
    ) -> None:
        super().__init__()
        if stride < 1 or encoder_kernel_size < stride or (encoder_kernel_size - stride) % 2:
            raise ValueError(
                "stride must be positive; encoder_kernel_size must be >= stride "
                "with an even difference"
            )
        padding = (encoder_kernel_size - stride) // 2
        self.stride = stride
        self.encoder = nn.Sequential(
            nn.Conv1d(3, encoder_channels, encoder_kernel_size, stride=stride, padding=padding),
            nn.GELU(),
        )
        self.temporal = TemporalConvNet(
            encoder_channels, num_channels, kernel_size, dropout,
            spatial_dropout=spatial_dropout,
        )
        self.decoder = nn.ConvTranspose1d(
            self.temporal.num_channels[-1], 3, encoder_kernel_size,
            stride=stride, padding=padding,
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 3 or signal.shape[1] != 3:
            raise ValueError("input must have shape [batch, 3, time]")
        if signal.shape[-1] % self.stride:
            raise ValueError(f"input length must be divisible by {self.stride}")
        return self.decoder(self.temporal(self.encoder(signal)))


# Standard TCN reference implementation license:
# https://github.com/locuslab/TCN/blob/master/LICENSE
# MIT License
# Copyright (c) 2018 CMU Locus Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
