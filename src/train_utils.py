# Entrenamiento local y evaluación.
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, recall_score

from .dataset import get_dataloader


# mu=0 → FedAvg; mu>0 → término proximal FedProx: (mu/2)*||w - w_global||^2
# Devuelve (model, avg_train_loss_ultima_epoca)
def local_train(model: nn.Module, dataset, epochs: int, lr: float, batch_size: int, device: torch.device, mu: float = 0.0):
    model = model.to(device)
    loader = get_dataloader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # w_global fijado al inicio — punto de anclaje del término proximal
    global_params = [p.detach().clone() for p in model.parameters()] if mu > 0 else None

    model.train()
    epoch_losses = []
    for _ in range(epochs):
        batch_losses = []
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
            batch_losses.append(loss.item())
        epoch_losses.append(float(np.mean(batch_losses)))

    return model, float(np.mean(epoch_losses))


# Devuelve {'auc', 'f1', 'sensitivity', 'val_loss'}
@torch.no_grad()
def evaluate(model: nn.Module, dataset, device: torch.device, batch_size: int = 128) -> dict:
    model.eval()
    model.to(device)
    loader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()

    probs_list, labels_list, losses = [], [], []
    for X, y in loader:
        X_dev, y_dev = X.to(device), y.to(device)
        logits = model(X_dev).squeeze(1)
        losses.append(criterion(logits, y_dev.float()).item())
        probs_list.extend(torch.sigmoid(logits).cpu().numpy())
        labels_list.extend(y.numpy())

    probs  = np.array(probs_list)
    labels = np.array(labels_list).astype(int)
    preds  = (probs >= 0.5).astype(int)

    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    f1  = float(f1_score(labels, preds, zero_division=0))
    sen = float(recall_score(labels, preds, pos_label=1, zero_division=0))

    return {"auc": float(auc), "f1": f1, "sensitivity": sen, "val_loss": float(np.mean(losses))}
