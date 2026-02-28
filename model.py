import torch
import torch.nn as nn
import torchvision.models as models


class PneumoniaResNet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(PneumoniaResNet18, self).__init__()
        
        if pretrained:
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.model = models.resnet18(weights=None)
        
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

