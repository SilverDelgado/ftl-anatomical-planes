import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_model(pretrained=True):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(512, 1)
    return model
