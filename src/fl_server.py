import time
from collections import OrderedDict

import numpy as np
import torch
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedProx
from flwr.common import Context, ndarrays_to_parameters

from .model import get_model
from .train_utils import evaluate


def model_to_ndarrays(model):
    arrays = []
    for val in model.state_dict().values():
        arrays.append(val.cpu().numpy())
    return arrays


def ndarrays_to_model(ndarrays, device):
    model = get_model(pretrained=False)
    orig = model.state_dict()
    state_dict = OrderedDict()
    for k, arr in zip(orig.keys(), ndarrays):
        state_dict[k] = torch.tensor(arr, dtype=orig[k].dtype)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device)


def weighted_average(metrics_list):
    total = 0
    for n, _ in metrics_list:
        total = total + n
    result = {}
    for key in metrics_list[0][1]:
        weighted = 0.0
        for n, m in metrics_list:
            weighted = weighted + n * float(m[key])
        result[key] = weighted / total
    return result


def make_server_app(global_test_ds, client_test_ds, rounds, algorithm, fit_config, device, state, mu=0.0, pretrained=True):
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
        "per_client_train_loss_per_epoch": [],
        "per_client_val_loss_per_epoch": [],
    })
    state["_t0"] = time.time()

    init_model = get_model(pretrained=pretrained)
    initial_parameters = ndarrays_to_parameters(model_to_ndarrays(init_model))

    def evaluate_fn(server_round, parameters, config):
        state["last_params"] = list(parameters)
        model = ndarrays_to_model(parameters, device)
        m = evaluate(model, global_test_ds, device)
        client_metrics = []
        for ds in client_test_ds:
            client_metrics.append(evaluate(model, ds, device))
        client_aucs = []
        for cm in client_metrics:
            if not np.isnan(cm["auc"]):
                client_aucs.append(cm["auc"])
        equity = float(np.std(client_aucs)) if client_aucs else float("nan")
        pc_auc, pc_f1, pc_sen, pc_spe, pc_acc, pc_loss = [], [], [], [], [], []
        for cm in client_metrics:
            pc_auc.append(cm["auc"])
            pc_f1.append(cm["f1"])
            pc_sen.append(cm["sensitivity"])
            pc_spe.append(cm["specificity"])
            pc_acc.append(cm["accuracy"])
            pc_loss.append(cm["val_loss"])
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
        h["per_client_auc"].append(pc_auc)
        h["per_client_f1"].append(pc_f1)
        h["per_client_sen"].append(pc_sen)
        h["per_client_spe"].append(pc_spe)
        h["per_client_acc"].append(pc_acc)
        h["per_client_loss"].append(pc_loss)
        return m["val_loss"], {
            "global_auc": m["auc"],
            "global_f1": m["f1"],
            "global_sen": m["sensitivity"],
            "global_spe": m["specificity"],
            "global_acc": m["accuracy"],
        }

    def fit_metrics_agg(metrics_list):
        n_total = 0
        for n, _ in metrics_list:
            n_total = n_total + n
        weighted_loss = 0.0
        for n, m in metrics_list:
            weighted_loss = weighted_loss + n * float(m["train_loss"])
        avg_train_loss = weighted_loss / n_total
        state["history"]["train_loss"].append(avg_train_loss)

        pc_train, pc_val = [], []
        for _, m in metrics_list:
            s_t = m.get("train_loss_per_epoch", "")
            s_v = m.get("val_loss_per_epoch", "")
            t_list = []
            for x in s_t.split(","):
                if x:
                    t_list.append(float(x))
            v_list = []
            for x in s_v.split(","):
                if x:
                    v_list.append(float(x))
            pc_train.append(t_list)
            pc_val.append(v_list)
        state["history"]["per_client_train_loss_per_epoch"].append(pc_train)
        state["history"]["per_client_val_loss_per_epoch"].append(pc_val)

        return {"train_loss": avg_train_loss}

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
    else:
        raise ValueError(f"algorithm desconocido: {algorithm}")

    def server_fn(context):
        return ServerAppComponents(
            strategy=strategy,
            config=ServerConfig(num_rounds=rounds),
        )

    return ServerApp(server_fn=server_fn)
