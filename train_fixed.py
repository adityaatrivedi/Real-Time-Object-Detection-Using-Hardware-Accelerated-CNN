import torch
import torch.nn as nn
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

# --- 1. DEFINE CUSTOM MODULES (With self.c2 fix) ---

class ConvBNActivation(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes
        self.c2 = out_planes  # <--- CRITICAL: Tells YOLO the output size

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

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNActivation(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNActivation(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            SqueezeExcitation(hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self.c2 = oup  # <--- CRITICAL: Tells YOLO the output size

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# --- 2. INJECT INTO YOLO ---
# This forces the parser to use OUR classes defined above
tasks.ConvBNActivation = ConvBNActivation
tasks.InvertedResidual = InvertedResidual

# --- 3. TRAIN ---
if __name__ == "__main__":
    # Load the model using the YAML from the previous step
    model = YOLO("yolov8-mobilenetv3.yaml")
    
    # Train
    model.train(data="coco8.yaml", epochs=100)