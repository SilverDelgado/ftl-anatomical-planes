import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, recall_score

from .dataset import get_dataloader

def local_train(model: nn.Module, dataset, epochs: int, lr: float, batch_size: int, device: torch.device, mu: float = 0.0):
    #entrena modelo durante epochs epocas
    model = model.to(device)
    loader = get_dataloader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss() #perdida de la capa final con 1 logit

    if mu > 0:
        global_params = []
        for p in model.parameters():
            global_params.append(p.detach().clone())
    else:
        global_params = None

    model.train()
    epoch_losses = []
    for _ in range(epochs):
        batch_losses = []
        for X, y in loader:
            # X: (batch_size, 3, 224, 224)
            # y: (batch_size,) labels 0.0 o 1.0
            X = X.to(device)
            y = y.to(device).unsqueeze(1)  # (batch_size,) -> (batch_size, 1) para que coincida con la salida del modelo
            optimizer.zero_grad()
            # model(X): (batch_size, 1) un logit por imagen
            # criterion loss espera ambos tensores con la misma forma (batch_size, 1)
            loss = criterion(model(X), y)
            if mu > 0:
                # FedProx penaliza desviarse del modelo global
                prox = sum(
                    ((p - pg.to(device)) ** 2).sum()
                    for p, pg in zip(model.parameters(), global_params)
                )
                loss = loss + (mu / 2.0) * prox
            loss.backward()   # calcula gradientes
            optimizer.step()  # actualiza pesos
            batch_losses.append(loss.item())  # .item() extrae el float del tensor
        epoch_losses.append(float(np.mean(batch_losses)))

    return model, float(np.mean(epoch_losses))


@torch.no_grad()
def evaluate(model: nn.Module, dataset, device: torch.device, batch_size: int = 128) -> dict:
    # devuelve auc, f1, sensitivity,val_loss
    model.eval()
    model.to(device)
    loader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()

    probs_list, labels_list, losses = [], [], []
    for X, y in loader:
        X_dev, y_dev = X.to(device), y.to(device)
        # model(X_dev): (batch_size, 1) -> (batch_size,)
        logits = model(X_dev).squeeze(1)
        losses.append(criterion(logits, y_dev.float()).item())
        # sigmoid convierte logit a prob [0,1] pal AUC
        probs_list.extend(torch.sigmoid(logits).cpu().numpy())
        labels_list.extend(y.numpy())

    probs  = np.array(probs_list)
    labels = np.array(labels_list).astype(int)
    preds  = (probs >= 0.5).astype(int)

    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    f1  = float(f1_score(labels, preds, zero_division=0))
    sen = float(recall_score(labels, preds, pos_label=1, zero_division=0))

    return {"auc": float(auc), "f1": f1, "sensitivity": sen, "val_loss": float(np.mean(losses))}
