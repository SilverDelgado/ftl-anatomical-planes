import json
import numpy as np
import torch
from pathlib import Path
from flwr.simulation import run_simulation

from .model import get_model
from .train_utils import evaluate as eval_fn
from .fl_client import make_client_app
from .fl_server import make_server_app, ndarrays_to_model

def run_federated(client_train_ds: list, global_test_ds, client_test_ds: list, rounds: int, local_epochs: int, lr: float, batch_size: int, device: torch.device, mu: float = 0.0, aggregation: str = "weighted", pretrained: bool = True, seed: int = 42, verbose: bool = False) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Entrenamiento federado con Flower simulation (FedAvg / FedProx).
    # Configuración que el servidor envia a cada cliente en cada ronda
    fit_config = {
        "local_epochs": local_epochs,
        "lr": lr,
        "batch_size": batch_size,
        # proximal_mu lo añade FedProx, en FedAvg lo lee el cliente como 0.0
        "proximal_mu": mu,
    }

    # state lo rellena fl server en la simulacion
    state: dict = {}

    client_app = make_client_app(client_train_ds, client_test_ds)
    server_app = make_server_app(
        global_test_ds=global_test_ds,
        rounds=rounds,
        mu=mu,
        aggregation=aggregation,
        fit_config=fit_config,
        device=device,
        state=state,
        pretrained=pretrained,
    )

    n_gpus = 0.25 if torch.cuda.is_available() else 0.0
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=len(client_train_ds),
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": n_gpus}},
        verbose_logging=verbose,
    )

    # eval por cliente con los parms de la ultima ronda
    last_params = state.get("last_params", [])
    final_g: dict = {}
    final_c: list = []
    equity = float("nan")

    if last_params:
        final_model = ndarrays_to_model(last_params, device)
        final_g = eval_fn(final_model, global_test_ds, device)
        final_c = []
        for ds in client_test_ds:
            final_c.append(eval_fn(final_model, ds, device))
        valid_aucs = []
        for m in final_c:
            if not np.isnan(m["auc"]):
                valid_aucs.append(m["auc"])
        equity = float(np.std(valid_aucs)) if valid_aucs else float("nan")

    # coste de comunicacion= 2 x numero params × 4 bytes (float32) por ronda
    n_params = sum(p.numel() for p in get_model(pretrained=False).parameters())
    comm_bytes = 2 * n_params * 4

    return {
        "history": state.get("history", {}),
        "final_global": final_g,
        "final_clients": final_c,
        "equity_auc": equity,
        "comm_bytes_per_round": comm_bytes,
        "n_params": n_params,
    }


def save_results(results: dict, path: Path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Resultados guardados en {path}")
