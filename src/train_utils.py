import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score

from .dataset import get_dataloader

def local_train(model, dataset, epochs, lr, batch_size, device, mu=0.0, val_dataset=None, grad_clip=1.0, weight_decay=0.0):
    model = model.to(device)
    loader = get_dataloader(dataset, batch_size=batch_size, shuffle=True) #creamos dataloader para dataset del cliente
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) #usamos Adam como optimizador de loss
    criterion = nn.BCEWithLogitsLoss() #perdida usamos binary cross entropy de logits porque es clasif binaria y modelo deuvelve logits

    if mu > 0: #termino proximal fedprox, si es mayor a 0 (se usa), guardamos copia de params globales para calcular el termino proximal
        global_params = []
        for p in model.parameters():
            global_params.append(p.detach().clone()) #detach para desconectar del calculo al hacer autograd porque es constante, clone pa copiar tensor
    else:
        global_params = None

    epoch_train_losses = []
    epoch_val_losses = []
    for _ in range(epochs):
        model.train()
        batch_losses = []
        for X, y in loader:
            X = X.to(device) #devuelve [Batch,1]
            y = y.to(device).unsqueeze(1) #viene [Batch], vamos a [Batch,1] para comparar bien luego (igualar dimension)
            optimizer.zero_grad() #gradiente a cero
            loss = criterion(model(X), y) #calculamos perdida entre salida logits del modelo y labels
            if mu > 0: #si esta activado termino proximaol de fedprox
                prox = 0.0
                for p, pg in zip(model.parameters(), global_params): #p: tensor pesos locales, pg: tensor pesos globales (guardados antes)
                    prox = prox + ((p - pg.to(device)) ** 2).sum() #calculamos distancia entre params locales y globales para termino promxial
                loss = loss + (mu / 2.0) * prox
            loss.backward() #calculamos gradientes de perdida
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip) #evitar exploding gradients
            optimizer.step() #actualizamos pesos
            batch_losses.append(loss.item()) #guardamos perdida de cada batch
        epoch_train_losses.append(float(np.mean(batch_losses))) #guardamos perdida de la epoch
        if val_dataset is not None:
            v = evaluate(model, val_dataset, device)
            epoch_val_losses.append(float(v["val_loss"])) #guardan val loss por epoch, lo usamos para ver si hay overfitting (train loss baja val loss sube)
            #por eso luego llamamos con test dataset , no necesitamos val porque no hacemos decisiones sobre parametros usando test

    return model, float(np.mean(epoch_train_losses)), epoch_train_losses, epoch_val_losses


@torch.no_grad() #no se calculan gradientes aqui (es solo evaluacion), ahorramos construir grafo autograd
def evaluate(model, dataset, device, batch_size=128, return_probs=False):
    model.eval() #cambiamos modelo a modo evaluacion
    model.to(device)
    loader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()

    probs_list, labels_list, losses = [], [], []
    for X, y in loader: #recorremos dataset
        X_dev, y_dev = X.to(device), y.to(device)
        logits = model(X_dev).squeeze(1) #[Batch,1] -> [Batch]
        losses.append(criterion(logits, y_dev.float()).item()) #calcula bce y guarda perdida de cada batch entero, .item() para sacar escalar del tensor
        probs_list.extend(torch.sigmoid(logits).cpu().numpy()) #pasa a probabilidades con sigmoid y mueve tensor a cpu y convierte a numpy para las metricas
        labels_list.extend(y.numpy()) #convierte tensor labels a numpy y guarda

    probs = np.array(probs_list)
    labels = np.array(labels_list).astype(int)

    finite = np.isfinite(probs) 
    if not finite.all(): #seguridad
        print(f"[WARN] problema de evaluate() probs no finitas y metricas a nan")
        return {
            "auc": float("nan"), "f1": float("nan"),
            "sensitivity": float("nan"), "specificity": float("nan"),
            "accuracy": float("nan"), "val_loss": float("nan"),
            **({"probs": probs.tolist(), "labels": labels.tolist()} if return_probs else {})
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
