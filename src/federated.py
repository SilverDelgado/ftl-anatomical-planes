"""Simulación de FedAvg y FedProx.

Algoritmo FedAvg (McMahan et al. 2017):
  w_global ← w_global + Σ_k (n_k/N) * Δw_k
  donde Δw_k = w_k_local - w_global

Algoritmo FedProx (Li et al. 2020):
  Igual que FedAvg pero cada cliente minimiza:
  L_k(w) + (μ/2) ||w - w_global||²
  El término proximal ancla los pesos locales al global,
  reduciendo el client drift causado por heterogeneidad.
"""
import copy
import json
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

from .model import get_model
from .train_utils import local_train, evaluate


def _fedavg_aggregate(global_model: torch.nn.Module, deltas: list, sizes: list):
    """Agregación ponderada por tamaño de dataset (FedAvg estándar)."""
    global_sd = global_model.state_dict()
    total = sum(sizes)
    new_sd = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_sd.items()}

    for delta, n in zip(deltas, sizes):
        w = n / total
        for k in new_sd:
            new_sd[k] += w * (global_sd[k].float() + delta[k].float())

    # Restaurar dtype original
    for k in global_sd:
        new_sd[k] = new_sd[k].to(global_sd[k].dtype)

    global_model.load_state_dict(new_sd)
    return global_model


def run_federated(
    client_train_ds: list,
    global_test_ds,
    client_test_ds: list,
    rounds: int,
    local_epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    mu: float = 0.0,
    pretrained: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Ejecuta entrenamiento federado (FedAvg si mu=0, FedProx si mu>0).

    Devuelve un dict con:
      - history: curvas por ronda
      - final: métricas finales (global + por cliente + equidad)
      - comm_bytes_per_round: coste de comunicación
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    global_model = get_model(pretrained=pretrained).to(device)

    # Coste de comunicación: envío + recepción de todos los parámetros (float32)
    n_params = sum(p.numel() for p in global_model.parameters())
    comm_bytes = 2 * n_params * 4  # bytes por ronda por cliente

    history = {
        "round": [], "global_auc": [], "global_f1": [], "global_sen": [],
        "client_aucs": [], "equity_auc": [],
    }

    for r in tqdm(range(rounds), desc=f"[seed={seed} mu={mu}] Rondas"):
        global_sd = copy.deepcopy(global_model.state_dict())
        deltas, sizes = [], []

        for ds in client_train_ds:
            local_model = copy.deepcopy(global_model)
            local_model = local_train(local_model, ds, local_epochs, lr, batch_size, device, mu=mu)

            local_sd = local_model.cpu().state_dict()
            delta = {k: (local_sd[k].float() - global_sd[k].float()).cpu() for k in global_sd}
            deltas.append(delta)
            sizes.append(len(ds))

        global_model = _fedavg_aggregate(global_model, deltas, sizes)

        # Evaluación
        g_metrics = evaluate(global_model, global_test_ds, device)
        c_metrics  = [evaluate(global_model, ds, device) for ds in client_test_ds]
        c_aucs     = [m["auc"] for m in c_metrics]

        history["round"].append(r + 1)
        history["global_auc"].append(g_metrics["auc"])
        history["global_f1"].append(g_metrics["f1"])
        history["global_sen"].append(g_metrics["sensitivity"])
        history["client_aucs"].append(c_aucs)
        history["equity_auc"].append(float(np.std(c_aucs)))

        if verbose:
            print(
                f"  R{r+1:3d} | AUC={g_metrics['auc']:.4f} "
                f"| equity={history['equity_auc'][-1]:.4f} "
                f"| clients={[f'{a:.3f}' for a in c_aucs]}"
            )

    final_g = evaluate(global_model, global_test_ds, device)
    final_c = [evaluate(global_model, ds, device) for ds in client_test_ds]

    return {
        "history": history,
        "final_global": final_g,
        "final_clients": final_c,
        "equity_auc": float(np.std([m["auc"] for m in final_c])),
        "comm_bytes_per_round": comm_bytes,
        "n_params": n_params,
    }


def save_results(results: dict, path: Path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Resultados guardados en {path}")
