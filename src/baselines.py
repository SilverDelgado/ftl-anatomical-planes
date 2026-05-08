# Baselines: local-only y centralizado
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import ConcatDataset

from .model import get_model
from .train_utils import local_train, evaluate


# Cada hospital entrena su propio modelo de forma aislada, sin colaboraciopn
def run_local_only(client_train_ds: list, global_test_ds, client_test_ds: list, rounds: int, local_epochs: int, lr: float, batch_size: int, device: torch.device, pretrained: bool = True, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # historial por hospital: lista de dicts con métricas por ronda
    client_histories = []
    models = []
    for _ in client_train_ds:
        client_histories.append({"round": [], "val_loss": [], "auc": [], "f1": [], "sensitivity": [], "train_loss": []})
        models.append(get_model(pretrained=pretrained).to(device))

    for r in range(1, rounds + 1):
        for k, (model, train_ds, test_ds, hist) in enumerate(zip(models, client_train_ds, client_test_ds, client_histories), start=1):
            model, train_loss = local_train(model, train_ds, local_epochs, lr, batch_size, device, mu=0.0)
            m = evaluate(model, test_ds, device)
            hist["round"].append(r)
            hist["train_loss"].append(train_loss)
            hist["val_loss"].append(m["val_loss"])
            hist["auc"].append(m["auc"])
            hist["f1"].append(m["f1"])
            hist["sensitivity"].append(m["sensitivity"])
        aucs = []
        for h in client_histories:
            aucs.append(h["auc"][-1])
        formatted = []
        for a in aucs:
            formatted.append(f"{a:.4f}")
        print(f"  Ronda {r}/{rounds} — AUCs: {formatted}", flush=True)

    # Las métricas finales ya están en hist (último elemento de cada lista),
    # las recogemos en lugar de re-evaluar el modelo
    final_results = []
    for k, hist in enumerate(client_histories, start=1):
        m = {"auc": hist["auc"][-1], "f1": hist["f1"][-1],
             "sensitivity": hist["sensitivity"][-1], "val_loss": hist["val_loss"][-1]}
        print(f"  Hospital {k}: AUC={m['auc']:.4f} F1={m['f1']:.4f} Sen={m['sensitivity']:.4f}")
        final_results.append({"hospital": k, **m})

    all_aucs = []
    for r in final_results:
        all_aucs.append(r["auc"])
    return {
        "clients": final_results,
        "histories": client_histories,
        "equity_auc": float(np.std(all_aucs)),
        "mean_auc": float(np.mean(all_aucs)),
    }


# Todos los datos juntos en un servidor central (centralizado) ignora privacidad obviamente
def run_centralized(client_train_ds: list, global_test_ds, client_test_ds: list, rounds: int, local_epochs: int, lr: float, batch_size: int, device: torch.device, pretrained: bool = True, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    all_train = ConcatDataset(client_train_ds)
    model = get_model(pretrained=pretrained).to(device)

    history = {"round": [], "val_loss": [], "auc": [], "f1": [], "sensitivity": [], "train_loss": []}

    for r in range(1, rounds + 1):
        model, train_loss = local_train(model, all_train, local_epochs, lr, batch_size, device, mu=0.0)
        m = evaluate(model, global_test_ds, device)
        history["round"].append(r)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(m["val_loss"])
        history["auc"].append(m["auc"])
        history["f1"].append(m["f1"])
        history["sensitivity"].append(m["sensitivity"])
        print(f"  Ronda {r}/{rounds} — Global AUC={m['auc']:.4f} F1={m['f1']:.4f}", flush=True)

    final_g = evaluate(model, global_test_ds, device)
    final_c = []
    for ds in client_test_ds:
        final_c.append(evaluate(model, ds, device))
    print(f"  Global AUC={final_g['auc']:.4f} F1={final_g['f1']:.4f}")

    client_aucs = []
    for m in final_c:
        client_aucs.append(m["auc"])
    return {
        "history": history,
        "final_global": final_g,
        "final_clients": final_c,
        "equity_auc": float(np.std(client_aucs)),
    }


def save_results(results: dict, path: Path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Resultados guardados en {path}")
