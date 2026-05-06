"""ResNet-18 adaptado a clasificación binaria EM vs control."""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def get_model(pretrained: bool = True) -> nn.Module:
    """ResNet-18 con ImageNet preentrenado + FC final → 1 logit binario.

    Se usa BCEWithLogitsLoss, no hace falta sigmoid aquí.
    Para obtener probabilidades: torch.sigmoid(model(x)).
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model   = resnet18(weights=weights)
    model.fc = nn.Linear(512, 1)
    return model


# --- Utilidades para serialización de pesos (necesarias en FedAvg) ---

def get_params(model: nn.Module) -> list:
    return [p.data.cpu().numpy().copy() for p in model.parameters()]


def set_params(model: nn.Module, params: list) -> None:
    import numpy as np
    for p, arr in zip(model.parameters(), params):
        p.data = torch.tensor(arr, dtype=p.dtype).to(p.device)
