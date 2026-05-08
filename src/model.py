import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# ResNet18 adaptado a clasificación binaria EM vs control:
# la última capa fully connected da 1 logit que combina con BCEWithLogitsLoss
def get_model(pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model   = resnet18(weights=weights)
    model.fc = nn.Linear(512, 1)
    return model
