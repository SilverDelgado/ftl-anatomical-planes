import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    HOSPITAL_TRAIN_CSVS, GLOBAL_TEST_CSV, DATA_ROOT, RESULTS_DIR,
    ROUNDS, LOCAL_EPOCHS, LR, BATCH_SIZE, IMG_SIZE, SEEDS,
    MU_FEDPROX, BETA_FEDAVGM, SERVER_LR,
)
from src.dataset import load_client_datasets, load_test_datasets
from src.baselines import run_local_only, run_centralized, save_results as save
from src.federated import run_federated

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {DEVICE}")

FEDERATED_EXPERIMENTS = {
    "fedavg":  {"algorithm": "fedavg",  "mu": 0.0,        "server_momentum": 0.0,          "server_lr": 1.0},
    "fedprox": {"algorithm": "fedprox", "mu": MU_FEDPROX, "server_momentum": 0.0,          "server_lr": 1.0},
    "fedavgm": {"algorithm": "fedavgm", "mu": 0.0,        "server_momentum": BETA_FEDAVGM, "server_lr": SERVER_LR},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["local", "centralized"] + list(FEDERATED_EXPERIMENTS), default=None)
    args = parser.parse_args()

    rounds = ROUNDS
    seeds = SEEDS

    print("Cargando datasets...")
    client_train = load_client_datasets(HOSPITAL_TRAIN_CSVS, DATA_ROOT, IMG_SIZE)
    global_test, client_test = load_test_datasets(GLOBAL_TEST_CSV, DATA_ROOT, IMG_SIZE)

    for k, ds in enumerate(client_train, 1):
        print(f"  Hospital {k} train: {len(ds)} slices")
    print(f"  Test global: {len(global_test)} slices")

    run = args.only

    models_dir = RESULTS_DIR / "models"

    if run in (None, "local"):
        print("\n=== Local-only ===")
        res = run_local_only(
            client_train, global_test, client_test,
            rounds=rounds, local_epochs=LOCAL_EPOCHS,
            lr=LR, batch_size=BATCH_SIZE, device=DEVICE, seed=seeds[0],
            weights_dir=models_dir,
        )
        save(res, RESULTS_DIR / "local_only.json")

    if run in (None, "centralized"):
        print("\n=== Centralizado ===")
        res = run_centralized(
            client_train, global_test, client_test,
            rounds=rounds, local_epochs=LOCAL_EPOCHS,
            lr=LR, batch_size=BATCH_SIZE, device=DEVICE, seed=seeds[0],
            weights_save_path=models_dir / "centralized.pt",
        )
        save(res, RESULTS_DIR / "centralized.json")

    for exp_name, exp_cfg in FEDERATED_EXPERIMENTS.items():
        if run not in (None, exp_name):
            continue
        print(f"\n=== {exp_name} ({exp_cfg}) ===")
        seed_results = []
        for seed in seeds:
            print(f"\n  Semilla {seed}")
            res = run_federated(
                client_train, global_test, client_test,
                rounds=rounds, local_epochs=LOCAL_EPOCHS,
                lr=LR, batch_size=BATCH_SIZE, device=DEVICE,
                seed=seed,
                weights_save_path=models_dir / f"{exp_name}_seed{seed}.pt",
                **exp_cfg,
            )
            seed_results.append(res)
        agg_stats = _aggregate_seeds(seed_results)
        save({"seeds": seed_results, "aggregated": agg_stats}, RESULTS_DIR / f"{exp_name}.json")

    print("Todos los experimentos completados. Resultados en results/")


def _aggregate_seeds(seed_results):
    def mean_std(vals):
        clean = [v for v in vals if not np.isnan(v)]
        return {"mean": float(np.mean(clean)) if clean else float("nan"),
                "std":  float(np.std(clean))  if clean else float("nan")}

    metric_keys = ["auc", "f1", "sensitivity", "specificity", "accuracy"]
    aggregated = {}
    for k in metric_keys:
        aggregated[f"global_{k}"] = mean_std([r["final_global"].get(k, float("nan")) for r in seed_results])

    aggregated["equity_auc"] = mean_std([r["equity_auc"] for r in seed_results])
    aggregated["comm_bytes_per_round"] = seed_results[0].get("comm_bytes_per_round", 0)
    aggregated["n_params"] = seed_results[0].get("n_params", 0)
    return aggregated


if __name__ == "__main__":
    main()
