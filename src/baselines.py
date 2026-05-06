# Baselines: local-only (cota inferior) y centralizado (cota superior).
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import ConcatDataset

from .model import get_model
from .train_utils import local_train, evaluate


# Cada hospital entrena su propio modelo de forma aislada, sin colaboración
def run_local_only(client_train_ds: list, global_test_ds, client_test_ds: list, rounds: int, local_epochs: int, lr: float, batch_size: int, device: torch.device, pretrained: bool = True, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    total_epochs = rounds * local_epochs
    results = []

    for k, (train_ds, test_ds) in enumerate(zip(client_train_ds, client_test_ds), start=1):
        model = get_model(pretrained=pretrained).to(device)
        model, _ = local_train(model, train_ds, total_epochs, lr, batch_size, device, mu=0.0)
        metrics = evaluate(model, test_ds, device)
        print(f"  Hospital {k}: AUC={metrics['auc']:.4f} F1={metrics['f1']:.4f} Sen={metrics['sensitivity']:.4f}")
        results.append({"hospital": k, **metrics})

    all_aucs = [r["auc"] for r in results]
    return {
        "clients": results,
        "equity_auc": float(np.std(all_aucs)),
        "mean_auc": float(np.mean(all_aucs)),
    }


# Todos los datos juntos en un servidor central; ignora privacidad
def run_centralized(client_train_ds: list, global_test_ds, client_test_ds: list, rounds: int, local_epochs: int, lr: float, batch_size: int, device: torch.device, pretrained: bool = True, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    all_train = ConcatDataset(client_train_ds)
    total_epochs = rounds * local_epochs

    model = get_model(pretrained=pretrained).to(device)
    model, _ = local_train(model, all_train, total_epochs, lr, batch_size, device, mu=0.0)

    final_g = evaluate(model, global_test_ds, device)
    final_c = [evaluate(model, ds, device) for ds in client_test_ds]
    print(f"  Global AUC={final_g['auc']:.4f} F1={final_g['f1']:.4f}")

    return {
        "final_global": final_g,
        "final_clients": final_c,
        "equity_auc": float(np.std([m["auc"] for m in final_c])),
    }


def save_results(results: dict, path: Path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Resultados guardados en {path}")
