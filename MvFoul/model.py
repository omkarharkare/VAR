import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class OffsideDetectionModel(nn.Module):
    def __init__(self):
        super(OffsideDetectionModel, self).__init__()
        # Load a pretrained ResNet50 model using the updated method
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze all layers except the last few
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Modify the final layer for binary classification (foul or not)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)
    
    def forward(self, x):
        return self.resnet(x)

# Instantiate the model
model = OffsideDetectionModel()