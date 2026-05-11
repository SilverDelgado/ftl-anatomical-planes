import torch.nn as nn
from torchvision.models import resnet18

def get_model():
    weights = None #por si usamos modelo pre entrenado o no
    model = resnet18(weights=weights)
    model.fc = nn.Linear(512, 1) #convertimos salida a una neurona con capa fully conected porque es clasificaicon binaria
    return model