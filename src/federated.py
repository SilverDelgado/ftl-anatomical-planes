import json
import numpy as np
import torch
from flwr.simulation import run_simulation

from .model import get_model
from .train_utils import evaluate as eval_fn
from .fl_client import make_client_app
from .fl_server import make_server_app, ndarrays_to_model


def run_federated(client_train_ds, global_test_ds, client_test_ds, rounds, local_epochs, lr, batch_size, device, algorithm="fedavg", mu=0.0, server_momentum=0.0, server_lr=1.0, pretrained=True, seed=42, verbose=False, weights_save_path=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    fit_config = {
        "local_epochs": local_epochs,
        "lr": lr,
        "batch_size": batch_size,
        "proximal_mu": mu,
        "seed": int(seed),
    }

    state = {}

    client_app = make_client_app(client_train_ds, client_test_ds)
    server_app = make_server_app(
        global_test_ds=global_test_ds,
        client_test_ds=client_test_ds,
        rounds=rounds,
        algorithm=algorithm,
        fit_config=fit_config,
        device=device,
        state=state,
        mu=mu,
        server_momentum=server_momentum,
        server_lr=server_lr,
        pretrained=pretrained,
    )

    n_gpus = 1.0 if torch.cuda.is_available() else 0.0
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=len(client_train_ds),
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": n_gpus}},
        verbose_logging=verbose,
    )

    last_params = state.get("last_params", [])
    final_g = {}
    final_c = []
    equity = float("nan")

    weights_path_str = None
    if last_params:
        final_model = ndarrays_to_model(last_params, device)
        final_g = eval_fn(final_model, global_test_ds, device, return_probs=True)
        final_c = [eval_fn(final_model, ds, device, return_probs=True) for ds in client_test_ds]
        valid_aucs = [m["auc"] for m in final_c if not np.isnan(m["auc"])]
        equity = float(np.std(valid_aucs)) if valid_aucs else float("nan")

        if weights_save_path is not None:
            from pathlib import Path
            weights_save_path = Path(weights_save_path)
            weights_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(final_model.state_dict(), weights_save_path)
            weights_path_str = str(weights_save_path)

    n_params = sum(p.numel() for p in get_model(pretrained=False).parameters())
    comm_bytes = 2 * n_params * 4

    return {
        "history": state.get("history", {}),
        "final_global": final_g,
        "final_clients": final_c,
        "equity_auc": equity,
        "comm_bytes_per_round": comm_bytes,
        "n_params": n_params,
        "algorithm": algorithm,
        "mu": mu,
        "server_momentum": server_momentum,
        "seed": seed,
        "weights_path": weights_path_str,
    }


def save_results(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Resultados guardados en {path}")
