import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score

from .dataset import get_dataloader

def local_train(model, dataset, epochs, lr, batch_size, device, mu=0.0, val_dataset=None, grad_clip=1.0, weight_decay=0.0):
    model = model.to(device)
    loader = get_dataloader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    if mu > 0:
        global_params = []
        for p in model.parameters():
            global_params.append(p.detach().clone())
    else:
        global_params = None

    epoch_train_losses = []
    epoch_val_losses = []
    for _ in range(epochs):
        model.train()
        batch_losses = []
        for X, y in loader:
            X = X.to(device)
            y = y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            if mu > 0:
                prox = 0.0
                for p, pg in zip(model.parameters(), global_params):
                    prox = prox + ((p - pg.to(device)) ** 2).sum()
                loss = loss + (mu / 2.0) * prox
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            batch_losses.append(loss.item())
        epoch_train_losses.append(float(np.mean(batch_losses)))
        if val_dataset is not None:
            v = evaluate(model, val_dataset, device)
            epoch_val_losses.append(float(v["val_loss"]))

    return model, float(np.mean(epoch_train_losses)), epoch_train_losses, epoch_val_losses


@torch.no_grad()
def evaluate(model, dataset, device, batch_size=128, return_probs=False):
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

    probs = np.array(probs_list)
    labels = np.array(labels_list).astype(int)

    finite = np.isfinite(probs)
    if not finite.all():
        n_finite = int(finite.sum())
        nan_count = len(probs) - n_finite
        print(f"  [WARN] evaluate(): {nan_count}/{len(probs)} probs no finitas; metricas a NaN")
        return {
            "auc": float("nan"), "f1": float("nan"),
            "sensitivity": float("nan"), "specificity": float("nan"),
            "accuracy": float("nan"), "val_loss": float("nan"),
            **({"probs": probs.tolist(), "labels": labels.tolist()} if return_probs else {}),
        }

    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    f1 = float(f1_score(labels, preds, zero_division=0))
    sen = float(recall_score(labels, preds, pos_label=1, zero_division=0))
    spe = float(recall_score(labels, preds, pos_label=0, zero_division=0))
    acc = float(accuracy_score(labels, preds))

    out = {
        "auc": float(auc),
        "f1": f1,
        "sensitivity": sen,
        "specificity": spe,
        "accuracy": acc,
        "val_loss": float(np.mean(losses)),
    }
    if return_probs:
        out["probs"] = probs.tolist()
        out["labels"] = labels.tolist()
    return out
