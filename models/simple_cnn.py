import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Simple CNN for square RGB image classification."""

    def __init__(self, num_classes: int = 10, input_size: int = 32):
        super().__init__()
        self.input_size = int(input_size)
        if self.input_size < 8:
            raise ValueError("input_size must be >= 8 for SimpleCNN")

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        pooled_size = self.input_size // 8
        if pooled_size < 1:
            raise ValueError("input_size is too small after three pooling layers")
        self.fc1 = nn.Linear(128 * pooled_size * pooled_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits
