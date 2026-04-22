# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralEncoder(nn.Module):
    """Simulates STFT using a learnable convolutional layer."""

    def __init__(self, input_length=600, window_size=64):
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

    def forward(self, x):
        """Input shape: (B, L, C) -> output magnitude spectrogram."""
        B, L, C = x.shape
        x = x.permute(0, 2, 1)  # (B, C, L)
        x = x.reshape(B * C, 1, L)  # (B*C, 1, L)
        x = self.conv(x)  # (B*C, 2*fft_bins, time_steps)
        x = x.view(B, C, 2 * self.fft_bins, -1)  # (B, C, 2*fft_bins, time_steps)

        real = x[:, :, 0::2, :]
        imag = x[:, :, 1::2, :]
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)  # (B, C, fft_bins, time_steps)
        return mag


class SINE(nn.Module):
    """Spectral Inception Encoder using multiple fixed or learnable convolutional layers."""

    def __init__(self, input_length=600, window_sizes=[32, 48, 64]):
        super().__init__()
        self.window_sizes = window_sizes
        self.stft_layers = nn.ModuleList()

        for window_size in window_sizes:
            stride = window_size // 2
            fft_bins = window_size // 2 + 1
            time_steps = math.ceil(input_length / stride) + 1
            padding = int(((time_steps - 1) * stride + window_size - input_length) / 2)

            conv_layer = nn.Conv1d(
                in_channels=1,
                out_channels=2 * fft_bins,
                kernel_size=window_size,
                stride=stride,
                padding=padding,
                bias=False
            )

            # Optional: initialize weights to simulate STFT with Hann window
            # t = torch.arange(window_size).float()
            # hann = torch.hann_window(window_size, periodic=True)
            # weight = torch.zeros(2 * fft_bins, window_size)
            # for i in range(fft_bins):
            #     omega = 2 * math.pi * i / window_size
            #     weight[2 * i] = torch.cos(omega * t) * hann
            #     weight[2 * i + 1] = torch.sin(-omega * t) * hann
            # conv_layer.weight.data = weight.unsqueeze(1)
            # conv_layer.weight.requires_grad = False

            self.stft_layers.append(conv_layer)

    def forward(self, x):
        B, L, C = x.shape
        x_reshaped = x.permute(0, 2, 1)  # (B, C, L)
        x_reshaped = x_reshaped.reshape(B * C, 1, L)  # (B*C, 1, L)

        multi_scale_outputs = []
        for conv_layer in self.stft_layers:
            x_conv = conv_layer(x_reshaped)  # (B*C, 2*fft_bins, time_steps)
            x_conv = x_conv.view(B, C, -1, x_conv.shape[-1])  # (B, C, 2*fft_bins, time_steps)

            real = x_conv[:, :, 0::2, :]
            imag = x_conv[:, :, 1::2, :]
            mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
            multi_scale_outputs.append(mag)

        return multi_scale_outputs


class ChannelAttention(nn.Module):
    """Channel attention module using average and max pooling."""

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
        B, C, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(B, C))
        max_out = self.fc(self.max_pool(x).view(B, C))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention.unsqueeze(2).unsqueeze(3)

class ChannelAttentionCNN(nn.Module):
    """CNN with channel attention for classification."""

    def __init__(self, num_classes, input_channels=12):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels

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


class SINCA_s(nn.Module):
    """Lightweight SINCA (~39K parameters)."""

    def __init__(self, num_classes=17, input_channels=12, input_length=600):
        super().__init__()
        window_sizes = [32, 48, 64]
        self.multi_scale_stft = SINE(input_length=input_length, window_sizes=window_sizes)

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels * 3, 32, kernel_size=(3, 3), dilation=2, padding=(2, 2)),
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
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        specs = self.multi_scale_stft(x)
        target_H, target_W = specs[-1].shape[2], specs[-1].shape[3]

        resized_specs = []
        for spec in specs:
                if spec.shape[2] != target_H or spec.shape[3] != target_W:
                    spec = F.interpolate(spec, size=(target_H, target_W), mode='nearest')
                resized_specs.append(spec)

        combined = torch.cat(resized_specs, dim=1)
        features = self.cnn(combined)
        output = self.classifier(features)
        return output


class SINCA_xs(nn.Module):
    """Extra-Light SINCA (~20K parameters)."""

    def __init__(self, num_classes=17, input_channels=12, input_length=600):
        super().__init__()
        window_sizes = [32, 48]
        self.multi_scale_stft = SINE(input_length=input_length, window_sizes=window_sizes)

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels * 2, 24, kernel_size=(3, 3), dilation=2, padding=(2, 2)),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            ChannelAttention(24),
            nn.MaxPool2d(2),

            nn.Conv2d(24, 48, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            ChannelAttention(48),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(48, num_classes)
        )

    def forward(self, x):
        specs = self.multi_scale_stft(x)
        target_H, target_W = specs[-1].shape[2], specs[-1].shape[3]

        resized_specs = []
        for spec in specs:
            if spec.shape[2] != target_H or spec.shape[3] != target_W:
                spec = F.interpolate(spec, size=(target_H, target_W), mode='nearest')
            resized_specs.append(spec)

        combined = torch.cat(resized_specs, dim=1)
        features = self.cnn(combined)
        output = self.classifier(features)
        return output


class SINCA_xxs(nn.Module):
    """Minimal SINCA (~17K parameters)."""

    def __init__(self, num_classes=17, input_channels=12, input_length=600):
        super().__init__()
        window_sizes = [48]
        self.multi_scale_stft = SINE(input_length=input_length, window_sizes=window_sizes)

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 24, kernel_size=(3, 3), dilation=2, padding=(2, 2)),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            ChannelAttention(24),
            nn.MaxPool2d(2),

            nn.Conv2d(24, 48, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            ChannelAttention(48),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(48, num_classes)
        )

    def forward(self, x):
        specs = self.multi_scale_stft(x)
        target_H, target_W = specs[-1].shape[2], specs[-1].shape[3]

        resized_specs = []
        for spec in specs:
            if spec.shape[2] != target_H or spec.shape[3] != target_W:
                spec = F.interpolate(spec, size=(target_H, target_W), mode='nearest')
            resized_specs.append(spec)

        combined = torch.cat(resized_specs, dim=1)
        features = self.cnn(combined)
        output = self.classifier(features)
        return output


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print(f"SINCA_s params: {count_parameters(SINCA_s())}")
    print(f"SINCA_xs params: {count_parameters(SINCA_xs())}")
    print(f"SINCA_xxs params: {count_parameters(SINCA_xxs())}")

    batch_size = 256
    seq_len = 600
    channels = 12
    num_classes = 17
    x = torch.randn(batch_size, seq_len, channels)

    # Test SINCA_xxs inference
    model_small = SINCA_xxs(num_classes=num_classes, input_channels=channels)
    out = model_small(x)
    print(f"Output shape: {out.shape}")

    # Latency test for SINCA_s on GPU (if available)
    if torch.cuda.is_available():
        model = SINCA_s(num_classes=num_classes, input_channels=channels).eval().cuda()
        dummy_input = x.cuda()
        repetitions = 10

        # Warm-up
        for _ in range(10):
            _ = model(dummy_input)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repetitions):
            _ = model(dummy_input)
        end.record()
        torch.cuda.synchronize()

        latency_ms = start.elapsed_time(end) / repetitions
        print(f'Average latency: {latency_ms:.3f} ms')
        print(f'FPS: {1000 / latency_ms:.2f}')
