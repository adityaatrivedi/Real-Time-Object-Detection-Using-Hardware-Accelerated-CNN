import torch
import torch.nn as nn

# 1. Standard Convolution + BatchNorm + Activation
class ConvBNActivation(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes

# 2. Squeeze-and-Excitation (SE) Block - Crucial for MobileNetV3
class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels, squeeze_factor=4):
        super().__init__()
        squeeze_channels = input_channels // squeeze_factor
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.scale = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        scale = self.fc1(x).mean((2, 3), keepdim=True)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        return x * self.scale(scale)

# 3. The Main Building Block: Inverted Residual
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNActivation(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNActivation(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # se
            SqueezeExcitation(hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)