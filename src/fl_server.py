import time
from collections import OrderedDict

import numpy as np
import torch
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedProx, FedAvgM
from flwr.common import Context, ndarrays_to_parameters

from .model import get_model
from .train_utils import evaluate


def model_to_ndarrays(model):
    return [val.cpu().numpy() for val in model.state_dict().values()]


def ndarrays_to_model(ndarrays, device):
    model = get_model(pretrained=False)
    orig = model.state_dict()
    state_dict = OrderedDict()
    for k, arr in zip(orig.keys(), ndarrays):
        state_dict[k] = torch.tensor(arr, dtype=orig[k].dtype)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device)


def weighted_average(metrics_list):
    total = sum(n for n, _ in metrics_list)
    result = {}
    for key in metrics_list[0][1]:
        result[key] = sum(n * float(m[key]) for n, m in metrics_list) / total
    return result


def make_server_app(global_test_ds, client_test_ds, rounds, algorithm, fit_config, device, state, mu=0.0, server_momentum=0.0, server_lr=1.0, pretrained=True):
    state.setdefault("history", {
        "round": [],
        "global_val_loss": [],
        "global_auc": [],
        "global_f1": [],
        "global_sen": [],
        "global_spe": [],
        "global_acc": [],
        "equity_auc": [],
        "wall_time": [],
        "train_loss": [],
        "per_client_auc": [],
        "per_client_f1": [],
        "per_client_sen": [],
        "per_client_spe": [],
        "per_client_acc": [],
        "per_client_loss": [],
    })
    state["_t0"] = time.time()

    init_model = get_model(pretrained=pretrained)
    initial_parameters = ndarrays_to_parameters(model_to_ndarrays(init_model))

    def evaluate_fn(server_round, parameters, config):
        state["last_params"] = list(parameters)
        model = ndarrays_to_model(parameters, device)
        m = evaluate(model, global_test_ds, device)
        client_metrics = [evaluate(model, ds, device) for ds in client_test_ds]
        client_aucs = [cm["auc"] for cm in client_metrics if not np.isnan(cm["auc"])]
        equity = float(np.std(client_aucs)) if client_aucs else float("nan")
        h = state["history"]
        h["round"].append(server_round)
        h["global_val_loss"].append(m["val_loss"])
        h["global_auc"].append(m["auc"])
        h["global_f1"].append(m["f1"])
        h["global_sen"].append(m["sensitivity"])
        h["global_spe"].append(m["specificity"])
        h["global_acc"].append(m["accuracy"])
        h["equity_auc"].append(equity)
        h["wall_time"].append(time.time() - state["_t0"])
        h["per_client_auc"].append([cm["auc"] for cm in client_metrics])
        h["per_client_f1"].append([cm["f1"] for cm in client_metrics])
        h["per_client_sen"].append([cm["sensitivity"] for cm in client_metrics])
        h["per_client_spe"].append([cm["specificity"] for cm in client_metrics])
        h["per_client_acc"].append([cm["accuracy"] for cm in client_metrics])
        h["per_client_loss"].append([cm["val_loss"] for cm in client_metrics])
        return m["val_loss"], {
            "global_auc": m["auc"],
            "global_f1": m["f1"],
            "global_sen": m["sensitivity"],
            "global_spe": m["specificity"],
            "global_acc": m["accuracy"],
        }

    def fit_metrics_agg(metrics_list):
        agg = weighted_average(metrics_list)
        state["history"]["train_loss"].append(agg.get("train_loss", float("nan")))
        return agg

    def on_fit_config_fn(server_round):
        cfg = dict(fit_config)
        cfg["server_round"] = int(server_round)
        return cfg

    common_kwargs = dict(
        initial_parameters=initial_parameters,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
        on_fit_config_fn=on_fit_config_fn,
        evaluate_fn=evaluate_fn,
        fit_metrics_aggregation_fn=fit_metrics_agg,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    if algorithm == "fedavg":
        strategy = FedAvg(**common_kwargs)
    elif algorithm == "fedprox":
        strategy = FedProx(proximal_mu=mu, **common_kwargs)
    elif algorithm == "fedavgm":
        strategy = FedAvgM(server_momentum=server_momentum, server_learning_rate=server_lr, **common_kwargs)
    else:
        raise ValueError(f"algorithm desconocido: {algorithm}")

    def server_fn(context):
        return ServerAppComponents(
            strategy=strategy,
            config=ServerConfig(num_rounds=rounds),
        )

    return ServerApp(server_fn=server_fn)
