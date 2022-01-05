import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_channels: int = 3, num_classes: int = 10) -> None:
        super(MLP, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(n_channels * 32 * 32, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x


def mlp(n_channels=3, num_classes=10) -> MLP:
    return MLP(n_channels=n_channels, num_classes=num_classes)
