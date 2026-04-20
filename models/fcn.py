import torch
import torch.nn as nn


class FCN(nn.Module): # architecture adapted from https://arxiv.org/pdf/1611.06455
    def __init__(self, in_channels=12, kernels=[8,5,3], hidden=[128,256,128]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden[0], kernel_size=kernels[0],padding='same'),
            nn.BatchNorm1d(hidden[0]),
            nn.ReLU(),

            nn.Conv1d(hidden[0], hidden[1], kernel_size=kernels[1],padding='same'),
            nn.BatchNorm1d(hidden[1]),
            nn.ReLU(),

            nn.Conv1d(hidden[1], hidden[2], kernel_size=kernels[2],padding='same'),
            nn.BatchNorm1d(hidden[2]),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden[2], 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.net(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.head(x)
        # x = self.sigmoid(x)
        return x.squeeze(-1)
