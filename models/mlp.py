import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(MLP, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(3072, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x


def mlp(num_classes=10) -> MLP:
    return MLP(num_classes=num_classes)
