# Flower client para el Proyecto 6.
# MSClient: NumPyClient que envuelve local_train y evaluate de train_utils.py
from collections import OrderedDict
import numpy as np
import torch
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from .model import get_model
from .train_utils import local_train, evaluate
from .config import LR, BATCH_SIZE


class MSClient(NumPyClient):
    def __init__(self, cid: int, train_ds, test_ds, device: torch.device):
        self.cid = cid
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.device = device
        # Sin pesos preentrenados: el servidor los distribuye en la ronda 0
        self.model = get_model(pretrained=False)

    def get_parameters(self, config):
        # Serializa state_dict completo (pesos + buffers BN)
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        orig = self.model.state_dict()
        keys = list(orig.keys())
        state_dict = OrderedDict(
            {k: torch.tensor(arr, dtype=orig[k].dtype) for k, arr in zip(keys, parameters)}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        mu = float(config.get("proximal_mu", 0.0))
        epochs = int(config.get("local_epochs", 5))
        lr = float(config.get("lr", LR))
        batch_size = int(config.get("batch_size", BATCH_SIZE))

        self.model, train_loss = local_train(
            self.model, self.train_ds, epochs, lr, batch_size, self.device, mu=mu
        )
        return self.get_parameters(config={}), len(self.train_ds), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        metrics = evaluate(self.model, self.test_ds, self.device)
        return metrics["val_loss"], len(self.test_ds), {
            "auc": metrics["auc"],
            "f1": metrics["f1"],
            "sensitivity": metrics["sensitivity"],
        }


def make_client_app(train_datasets: list, test_datasets: list) -> ClientApp:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def client_fn(context: Context):
        cid = int(context.node_config["partition-id"])
        return MSClient(cid, train_datasets[cid], test_datasets[cid], device).to_client()

    return ClientApp(client_fn=client_fn)
