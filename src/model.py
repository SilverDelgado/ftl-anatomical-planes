# ResNet-18 adaptado a clasificación binaria EM vs control.
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


# FC final → 1 logit; usar BCEWithLogitsLoss. Para probabilidades: torch.sigmoid(model(x))
def get_model(pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model   = resnet18(weights=weights)
    model.fc = nn.Linear(512, 1)
    return model


def get_params(model: nn.Module) -> list:
    return [p.data.cpu().numpy().copy() for p in model.parameters()]


def set_params(model: nn.Module, params: list) -> None:
    import numpy as np
    for p, arr in zip(model.parameters(), params):
        p.data = torch.tensor(arr, dtype=p.dtype).to(p.device)
