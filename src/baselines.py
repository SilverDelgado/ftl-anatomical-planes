import json
import time
import numpy as np
import torch
from torch.utils.data import ConcatDataset

from .model import get_model
from .train_utils import local_train, evaluate


def _empty_history():
    return {
        "round": [], "val_loss": [],
        "auc": [], "f1": [], "sensitivity": [], "specificity": [], "accuracy": [],
        "train_loss": [], "wall_time": [],
    }


def _empty_central_history():
    return {
        "round": [], "val_loss": [],
        "auc": [], "f1": [], "sensitivity": [], "specificity": [], "accuracy": [],
        "train_loss": [], "wall_time": [],
        "per_client_auc": [], "per_client_f1": [], "per_client_sen": [],
        "per_client_spe": [], "per_client_acc": [], "per_client_loss": [],
    }


def run_local_only(client_train_ds, global_test_ds, client_test_ds, rounds, local_epochs, lr, batch_size, device, pretrained=True, seed=42, weights_dir=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    client_histories = [_empty_history() for _ in client_train_ds]
    models = [get_model(pretrained=pretrained).to(device) for _ in client_train_ds]

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
            hist["specificity"].append(m["specificity"])
            hist["accuracy"].append(m["accuracy"])
            hist["wall_time"].append(time.time() - t0)
        aucs = [h["auc"][-1] for h in client_histories]
        print(f"  Ronda {r}/{rounds} - AUCs: {[f'{a:.4f}' for a in aucs]}", flush=True)

    final_results = []
    weights_paths = []
    if weights_dir is not None:
        from pathlib import Path
        weights_dir = Path(weights_dir)
        weights_dir.mkdir(parents=True, exist_ok=True)

    for k, (model, test_ds) in enumerate(zip(models, client_test_ds), start=1):
        m = evaluate(model, test_ds, device, return_probs=True)
        print(f"  Hospital {k}: AUC={m['auc']:.4f} F1={m['f1']:.4f} Acc={m['accuracy']:.4f}")
        if weights_dir is not None:
            wp = weights_dir / f"local_only_h{k}.pt"
            torch.save(model.state_dict(), wp)
            weights_paths.append(str(wp))
        final_results.append({"hospital": k, **m})

    all_aucs = [r["auc"] for r in final_results]
    return {
        "clients": final_results,
        "histories": client_histories,
        "equity_auc": float(np.std(all_aucs)),
        "mean_auc": float(np.mean(all_aucs)),
        "weights_paths": weights_paths,
    }


def run_centralized(client_train_ds, global_test_ds, client_test_ds, rounds, local_epochs, lr, batch_size, device, pretrained=True, seed=42, weights_save_path=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    all_train = ConcatDataset(client_train_ds)
    model = get_model(pretrained=pretrained).to(device)
    history = _empty_central_history()

    for r in range(1, rounds + 1):
        model, train_loss = local_train(model, all_train, local_epochs, lr, batch_size, device, mu=0.0)
        m = evaluate(model, global_test_ds, device)
        cm_per_client = [evaluate(model, ds, device) for ds in client_test_ds]
        history["round"].append(r)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(m["val_loss"])
        history["auc"].append(m["auc"])
        history["f1"].append(m["f1"])
        history["sensitivity"].append(m["sensitivity"])
        history["specificity"].append(m["specificity"])
        history["accuracy"].append(m["accuracy"])
        history["wall_time"].append(time.time() - t0)
        history["per_client_auc"].append([c["auc"] for c in cm_per_client])
        history["per_client_f1"].append([c["f1"] for c in cm_per_client])
        history["per_client_sen"].append([c["sensitivity"] for c in cm_per_client])
        history["per_client_spe"].append([c["specificity"] for c in cm_per_client])
        history["per_client_acc"].append([c["accuracy"] for c in cm_per_client])
        history["per_client_loss"].append([c["val_loss"] for c in cm_per_client])
        print(f"  Ronda {r}/{rounds} - Global AUC={m['auc']:.4f} F1={m['f1']:.4f}", flush=True)

    final_g = evaluate(model, global_test_ds, device, return_probs=True)
    final_c = [evaluate(model, ds, device, return_probs=True) for ds in client_test_ds]
    print(f"  Global AUC={final_g['auc']:.4f} F1={final_g['f1']:.4f}")

    weights_path_str = None
    if weights_save_path is not None:
        from pathlib import Path
        weights_save_path = Path(weights_save_path)
        weights_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), weights_save_path)
        weights_path_str = str(weights_save_path)

    client_aucs = [m["auc"] for m in final_c]
    return {
        "history": history,
        "final_global": final_g,
        "final_clients": final_c,
        "equity_auc": float(np.std(client_aucs)),
        "weights_path": weights_path_str,
    }


def save_results(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Resultados guardados en {path}")
