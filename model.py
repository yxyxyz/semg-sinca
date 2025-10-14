# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn


###############################################################################
#                       STFT Imitation Network                                #
###############################################################################
class STFTImitNet(nn.Module):
    def __init__(self, input_length=600, window_size=48):
        super().__init__()
        self.window_size = window_size
        self.stride = window_size // 2
        self.fft_bins = window_size // 2 + 1
        self.time_steps = math.ceil(input_length / self.stride) + 1
        self.padding = int(((self.time_steps - 1) * self.stride + window_size - input_length) / 2)

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=2 * self.fft_bins,
            kernel_size=self.window_size,
            stride=self.stride,
            padding=self.padding,
            bias=False
        )
        # self._init_weights()

    def _init_weights(self):
        """Initialize weights to simulate STFT operation"""
        t = torch.arange(self.window_size).float()
        hann = torch.hann_window(self.window_size, periodic=True)
        weight = torch.zeros(2 * self.fft_bins, self.window_size)

        for i in range(self.fft_bins):
            omega = 2 * math.pi * i / self.window_size
            weight[2 * i] = torch.cos(omega * t) * hann
            weight[2 * i + 1] = torch.sin(-omega * t) * hann

        self.conv.weight.data = weight.unsqueeze(1)

    def forward(self, x):
        """Input size (B, L, C) = (batch, seq_len, channels)"""
        B, L, C = x.shape

        # Rearrange dimensions and flatten channels
        x = x.permute(0, 2, 1)  # (B, C, L)
        x = x.reshape(B * C, 1, L)  # (B*C, 1, seq_len)
        x = self.conv(x)  # (B*C, 2*fft_bins, time_steps)
        x = x.view(B, C, 2 * self.fft_bins, -1)  # (B, C, 2*fft_bins, time_steps)

        real = x[:, :, 0::2, :]
        imag = x[:, :, 1::2, :]

        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)  # (B, C, fft_bins, time_steps)
        return mag


###############################################################################
#                      Channel Attention Module                               #
###############################################################################
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [B, C, H, W]
        B, C, _, _ = x.size()

        avg_out = self.fc(self.avg_pool(x).view(B, C))
        max_out = self.fc(self.max_pool(x).view(B, C))

        # Combine average and max pooling attention weights
        attention = self.sigmoid(avg_out + max_out)
        return x * attention.unsqueeze(2).unsqueeze(3)


###############################################################################
#                       Channel Attention CNN                                 #
###############################################################################
class ChannelAttentionCNN(nn.Module):
    def __init__(self, num_classes, input_channels=12):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels

        # Convolutional layers with batch normalization
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ChannelAttention(32),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ChannelAttention(64),
            nn.MaxPool2d((2, 2)),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        cnn_feat = self.cnn(x)
        out = self.classifier(cnn_feat)
        return out


###############################################################################
#                               SINCA Model                                   #
###############################################################################
class SINCA(nn.Module):
    def __init__(self, num_classes=17, input_channels=12, input_length=600, window_size=48):
        super().__init__()
        self.stft_layer = STFTImitNet(input_length=input_length, window_size=window_size)

        # Enhanced CNN architecture
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ChannelAttention(32),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ChannelAttention(64),
            nn.MaxPool2d((2, 2)),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        spectrogram = self.stft_layer(x)
        cnn_feat = self.cnn(spectrogram)
        out = self.classifier(cnn_feat)
        return out


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    model = SINCA()
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params}")
