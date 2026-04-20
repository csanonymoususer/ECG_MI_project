import torch
import torch.nn as nn


class ResNetBlock(nn.Module): # architecture adapted from https://arxiv.org/pdf/1611.06455
    def __init__(self, in_channels, out_channels, kernels=[8,5,3]):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernels[0], padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernels[1], padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernels[2], padding='same'),
            nn.BatchNorm1d(out_channels)
        )

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )


    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.net(x)
        x = x + shortcut
        x = torch.relu(x)
        return x
    


class ResNet(nn.Module):
    def __init__(self, in_channels=12, kernels=[8,5,3], hidden=[64,128,128]):
        super().__init__()

        self.block1 = ResNetBlock(in_channels, hidden[0], kernels=kernels)
        self.block2 = ResNetBlock(hidden[0], hidden[1], kernels=kernels)
        self.block3 = ResNetBlock(hidden[1], hidden[2], kernels=kernels)
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden[2], 1)


    def forward(self, x):
        x = self.block1(x)
        x = self.drop1(x)
        x = self.block2(x)
        x = self.drop2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.head(x)
        return x.squeeze(-1)
