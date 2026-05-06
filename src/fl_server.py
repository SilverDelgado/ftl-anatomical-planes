# Flower server: estrategias FedAvg y FedProx con evaluate_fn global.
# Soporta agregación ponderada (estándar) y simple (igual peso por cliente).
from collections import OrderedDict
import numpy as np
import torch
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedProx
from flwr.common import Context, FitRes, ndarrays_to_parameters

from .model import get_model
from .train_utils import evaluate


# ── Helpers de serialización ───────────────────────────────────────────────────

def model_to_ndarrays(model):
    return [val.cpu().numpy() for val in model.state_dict().values()]

def ndarrays_to_model(ndarrays, device):
    model = get_model(pretrained=False)
    orig = model.state_dict()
    keys = list(orig.keys())
    state_dict = OrderedDict(
        {k: torch.tensor(arr, dtype=orig[k].dtype) for k, arr in zip(keys, ndarrays)}
    )
    model.load_state_dict(state_dict, strict=True)
    return model.to(device)


# ── Funciones de agregación de métricas ───────────────────────────────────────

def weighted_average(metrics_list: list) -> dict:
    # Promedio ponderado por n_k: da más peso a hospitales con más datos
    total = sum(n for n, _ in metrics_list)
    result = {}
    for key in metrics_list[0][1]:
        result[key] = sum(n * float(m[key]) for n, m in metrics_list) / total
    return result


def simple_average(metrics_list: list) -> dict:
    # Promedio simple: todos los hospitales pesan igual (evita que H4 domine)
    n = len(metrics_list)
    result = {}
    for key in metrics_list[0][1]:
        result[key] = sum(float(m[key]) for _, m in metrics_list) / n
    return result


# ── Estrategias con agregación simple ─────────────────────────────────────────

class FedAvgSimple(FedAvg):
    # Sobreescribe num_examples=1 para forzar pesos uniformes en la agregación
    def aggregate_fit(self, server_round, results, failures):
        results_eq = [
            (proxy, FitRes(
                status=r.status, parameters=r.parameters,
                num_examples=1, metrics=r.metrics,
            ))
            for proxy, r in results
        ]
        return super().aggregate_fit(server_round, results_eq, failures)


class FedProxSimple(FedProx):
    def aggregate_fit(self, server_round, results, failures):
        results_eq = [
            (proxy, FitRes(
                status=r.status, parameters=r.parameters,
                num_examples=1, metrics=r.metrics,
            ))
            for proxy, r in results
        ]
        return super().aggregate_fit(server_round, results_eq, failures)


# ── Fábrica principal ──────────────────────────────────────────────────────────

# state: dict mutable compartido con federated.py; fl_server lo rellena durante la simulación
# aggregation: 'weighted' (estándar FedAvg) | 'simple' (igual peso por cliente, evita que H4 domine)
def make_server_app(global_test_ds, rounds: int, mu: float, aggregation: str, fit_config: dict, device: torch.device, state: dict, pretrained: bool = True) -> ServerApp:
    state.setdefault("history", {
        "round": [], "global_val_loss": [],
        "global_auc": [], "global_f1": [], "global_sen": [],
        "train_loss": [],
    })

    # Parámetros iniciales: ResNet-18 preentrenado en ImageNet
    init_model = get_model(pretrained=pretrained)
    initial_parameters = ndarrays_to_parameters(model_to_ndarrays(init_model))

    metrics_agg = weighted_average if aggregation == "weighted" else simple_average

    def evaluate_fn(server_round, parameters, config):
        # Corre en el proceso principal → puede modificar state
        state["last_params"] = list(parameters)
        model = ndarrays_to_model(parameters, device)
        m = evaluate(model, global_test_ds, device)
        h = state["history"]
        h["round"].append(server_round)
        h["global_val_loss"].append(m["val_loss"])
        h["global_auc"].append(m["auc"])
        h["global_f1"].append(m["f1"])
        h["global_sen"].append(m["sensitivity"])
        return m["val_loss"], {"global_auc": m["auc"], "global_f1": m["f1"], "global_sen": m["sensitivity"]}

    def fit_metrics_agg(metrics_list):
        agg = metrics_agg(metrics_list)
        state["history"]["train_loss"].append(agg.get("train_loss", float("nan")))
        return agg

    def on_fit_config_fn(server_round):
        return fit_config

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
        evaluate_metrics_aggregation_fn=metrics_agg,
    )

    if mu == 0.0:
        strategy = FedAvg(**common_kwargs) if aggregation == "weighted" else FedAvgSimple(**common_kwargs)
    else:
        strategy = (FedProx if aggregation == "weighted" else FedProxSimple)(
            proximal_mu=mu, **common_kwargs
        )

    def server_fn(context: Context) -> ServerAppComponents:
        return ServerAppComponents(
            strategy=strategy,
            config=ServerConfig(num_rounds=rounds),
        )

    return ServerApp(server_fn=server_fn)
