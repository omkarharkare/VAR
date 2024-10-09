import torch
import torch.nn as nn

class FoulClassificationModel(nn.Module):
    def __init__(self, input_shape=(16, 224, 224, 3)):
        super(FoulClassificationModel, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.fc1 = None  # We'll define this in the forward pass
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], 256).to(x.device)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x