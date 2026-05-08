from collections import OrderedDict
import numpy as np
import torch
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedProx
from flwr.common import Context, FitRes, ndarrays_to_parameters

from .model import get_model
from .train_utils import evaluate
#helpers para convertir modelo entre pytorch y numpy arrays
def model_to_ndarrays(model):
    arrays = []
    for val in model.state_dict().values():
        arrays.append(val.cpu().numpy())
    return arrays

def ndarrays_to_model(ndarrays, device):
    model = get_model(pretrained=False)
    orig = model.state_dict()
    keys = list(orig.keys())
    state_dict = OrderedDict()
    for k, arr in zip(keys, ndarrays):
        state_dict[k] = torch.tensor(arr, dtype=orig[k].dtype)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device)

#agregacion de metricas
def weighted_average(metrics_list: list) -> dict:
    # avg ponderado (> peso a hospitales con + datos)
    total = sum(n for n, _ in metrics_list)
    result = {}
    for key in metrics_list[0][1]:
        result[key] = sum(n * float(m[key]) for n, m in metrics_list) / total
    return result


def simple_average(metrics_list: list) -> dict:
    # avg simple(todos los hospitales pesan iguala, no importa numero de muestras)
    n = len(metrics_list)
    result = {}
    for key in metrics_list[0][1]:
        result[key] = sum(float(m[key]) for _, m in metrics_list) / n
    return result

class FedAvgSimple(FedAvg):
    # num_examples=1 para forzar pesos uniformes en la agregación
    def aggregate_fit(self, server_round, results, failures):
        results_eq = []
        for proxy, r in results:
            results_eq.append((proxy, FitRes(
                status=r.status, parameters=r.parameters,
                num_examples=1, metrics=r.metrics,
            )))
        return super().aggregate_fit(server_round, results_eq, failures)


class FedProxSimple(FedProx):
    def aggregate_fit(self, server_round, results, failures):
        results_eq = []
        for proxy, r in results:
            results_eq.append((proxy, FitRes(
                status=r.status, parameters=r.parameters,
                num_examples=1, metrics=r.metrics,
            )))
        return super().aggregate_fit(server_round, results_eq, failures)

# aggregation: weighted, simple
def make_server_app(global_test_ds, client_test_ds: list, rounds: int, mu: float, aggregation: str, fit_config: dict, device: torch.device, state: dict, pretrained: bool = True) -> ServerApp:
    state.setdefault("history", {
        "round": [], "global_val_loss": [],
        "global_auc": [], "global_f1": [], "global_sen": [],
        # equity_auc por ronda: std de los AUC por hospital. Lo necesita
        # notebooks/analysis.ipynb (celda 8) para la curva de equidad.
        "equity_auc": [],
        "train_loss": [],
    })

    # Params iniciales: ResNet18 preentrenado en ImageNet
    init_model = get_model(pretrained=pretrained)
    initial_parameters = ndarrays_to_parameters(model_to_ndarrays(init_model))

    metrics_agg = weighted_average if aggregation == "weighted" else simple_average

    def evaluate_fn(server_round, parameters, config):
        # Corre en el proceso principal -> puede modificar state
        state["last_params"] = list(parameters)
        model = ndarrays_to_model(parameters, device)
        m = evaluate(model, global_test_ds, device)
        # Equidad por ronda: evaluamos el modelo agregado en el test de cada
        # hospital y guardamos la std de los AUCs (excluyendo NaN)
        client_aucs = []
        for ds in client_test_ds:
            cm = evaluate(model, ds, device)
            if not np.isnan(cm["auc"]):
                client_aucs.append(cm["auc"])
        equity = float(np.std(client_aucs)) if client_aucs else float("nan")
        h = state["history"]
        h["round"].append(server_round)
        h["global_val_loss"].append(m["val_loss"])
        h["global_auc"].append(m["auc"])
        h["global_f1"].append(m["f1"])
        h["global_sen"].append(m["sensitivity"])
        h["equity_auc"].append(equity)
        return m["val_loss"], {"global_auc": m["auc"], "global_f1": m["f1"], "global_sen": m["sensitivity"]}

    def fit_metrics_agg(metrics_list):
        agg = metrics_agg(metrics_list)
        state["history"]["train_loss"].append(agg.get("train_loss", float("nan")))
        return agg

    def on_fit_config_fn(server_round):
        # Inyectamos server_round para que el cliente pueda mezclarlo en su seed
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
