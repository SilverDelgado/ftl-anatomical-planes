import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# ResNet18 adaptado a clasificacion binaria EM vs control
# capa fully connected final que da 1 logit
def get_model(pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model   = resnet18(weights=weights)
    model.fc = nn.Linear(512, 1) #capa fully connected final con 1 salida
    return model


def get_params(model: nn.Module) -> list:
    params = []
    for p in model.parameters():
        params.append(p.data.cpu().numpy().copy())
    return params

def set_params(model: nn.Module, params: list) -> None:
    for p, arr in zip(model.parameters(), params):
        p.data = torch.tensor(arr, dtype=p.dtype).to(p.device)
