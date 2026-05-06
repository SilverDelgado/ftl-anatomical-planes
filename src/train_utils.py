"""Entrenamiento local y evaluación."""
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, recall_score


def local_train(
    model: nn.Module,
    dataset,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    mu: float = 0.0,
) -> nn.Module:
    """Entrena el modelo localmente.

    mu=0   → entrenamiento estándar (FedAvg).
    mu>0   → añade término proximal FedProx: (mu/2)*||w - w_global||^2.
              w_global se fija como los pesos al inicio de esta función.
    """
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Captura estado global para FedProx
    global_params = [p.detach().clone() for p in model.parameters()] if mu > 0 else None

    model.train()
    for _ in range(epochs):
        for X, y in loader:
            X, y = X.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(X), y)

            if mu > 0:
                prox = sum(
                    ((p - pg.to(device)) ** 2).sum()
                    for p, pg in zip(model.parameters(), global_params)
                )
                loss = loss + (mu / 2.0) * prox

            loss.backward()
            optimizer.step()

    return model


@torch.no_grad()
def evaluate(model: nn.Module, dataset, device: torch.device, batch_size: int = 128) -> dict:
    """Calcula AUC-ROC, F1 y sensibilidad (recall clase MS=1)."""
    model.eval()
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    probs_list, labels_list = [], []
    for X, y in loader:
        logits = model(X.to(device)).squeeze(1).cpu()
        probs_list.extend(torch.sigmoid(logits).numpy())
        labels_list.extend(y.numpy())

    probs  = np.array(probs_list)
    labels = np.array(labels_list).astype(int)
    preds  = (probs >= 0.5).astype(int)

    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    f1  = float(f1_score(labels, preds, zero_division=0))
    sen = float(recall_score(labels, preds, pos_label=1, zero_division=0))

    return {"auc": float(auc), "f1": f1, "sensitivity": sen}
