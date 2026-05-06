# Ejecuta todos los experimentos del Proyecto 6 y guarda resultados en results/.
# Uso: uv run python run_all.py [--dry-run] [--only local|centralized|fedavg|fedprox|fedavg_simple|fedprox_simple]
# Experimentos federados se repiten con 3 semillas; baselines con 1 semilla.
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    HOSPITAL_TRAIN_CSVS, GLOBAL_TEST_CSV, DATA_ROOT, RESULTS_DIR,
    ROUNDS, LOCAL_EPOCHS, LR, BATCH_SIZE, IMG_SIZE, SEEDS, MU_FEDPROX,
)
from src.dataset import load_client_datasets, load_test_datasets
from src.baselines import run_local_only, run_centralized, save_results as save
from src.federated import run_federated

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {DEVICE}")

FEDERATED_EXPERIMENTS = {
    "fedavg":        {"mu": 0.0,        "aggregation": "weighted"},
    "fedprox":       {"mu": MU_FEDPROX, "aggregation": "weighted"},
    "fedavg_simple": {"mu": 0.0,        "aggregation": "simple"},
    "fedprox_simple":{"mu": MU_FEDPROX, "aggregation": "simple"},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Solo 2 rondas, 1 semilla")
    parser.add_argument("--only", choices=["local", "centralized"] + list(FEDERATED_EXPERIMENTS), default=None)
    args = parser.parse_args()

    rounds = 2 if args.dry_run else ROUNDS
    seeds  = [SEEDS[0]] if args.dry_run else SEEDS

    print("Cargando datasets...")
    client_train = load_client_datasets(HOSPITAL_TRAIN_CSVS, DATA_ROOT, IMG_SIZE)
    global_test, client_test = load_test_datasets(GLOBAL_TEST_CSV, DATA_ROOT, IMG_SIZE)

    for k, ds in enumerate(client_train, 1):
        print(f"  Hospital {k} train: {len(ds)} slices")
    print(f"  Test global: {len(global_test)} slices")

    run = args.only

    # --- 1. Local-only ---
    if run in (None, "local"):
        print("\n=== Local-only (cota inferior) ===")
        res = run_local_only(
            client_train, global_test, client_test,
            rounds=rounds, local_epochs=LOCAL_EPOCHS,
            lr=LR, batch_size=BATCH_SIZE, device=DEVICE, seed=seeds[0],
        )
        save(res, RESULTS_DIR / "local_only.json")

    # --- 2. Centralizado ---
    if run in (None, "centralized"):
        print("\n=== Centralizado (cota superior) ===")
        res = run_centralized(
            client_train, global_test, client_test,
            rounds=rounds, local_epochs=LOCAL_EPOCHS,
            lr=LR, batch_size=BATCH_SIZE, device=DEVICE, seed=seeds[0],
        )
        save(res, RESULTS_DIR / "centralized.json")

    # --- 3-6. Experimentos federados ---
    for exp_name, exp_cfg in FEDERATED_EXPERIMENTS.items():
        if run not in (None, exp_name):
            continue
        mu  = exp_cfg["mu"]
        agg = exp_cfg["aggregation"]
        print(f"\n=== {exp_name} (mu={mu}, agregación={agg}) ===")
        seed_results = []
        for seed in seeds:
            print(f"\n  Semilla {seed}")
            res = run_federated(
                client_train, global_test, client_test,
                rounds=rounds, local_epochs=LOCAL_EPOCHS,
                lr=LR, batch_size=BATCH_SIZE, device=DEVICE,
                mu=mu, aggregation=agg, seed=seed,
            )
            seed_results.append(res)
        agg_stats = _aggregate_seeds(seed_results)
        save({"seeds": seed_results, "aggregated": agg_stats}, RESULTS_DIR / f"{exp_name}.json")

    print("\nTodos los experimentos completados. Resultados en results/")


def _aggregate_seeds(seed_results: list) -> dict:
    # Promedia métricas finales sobre semillas
    def mean_std(vals):
        clean = [v for v in vals if not np.isnan(v)]
        return {"mean": float(np.mean(clean)) if clean else float("nan"),
                "std":  float(np.std(clean))  if clean else float("nan")}

    global_aucs = [r["final_global"].get("auc", float("nan")) for r in seed_results]
    global_f1s  = [r["final_global"].get("f1",  float("nan")) for r in seed_results]
    global_sens = [r["final_global"].get("sensitivity", float("nan")) for r in seed_results]
    equities    = [r["equity_auc"] for r in seed_results]

    return {
        "global_auc": mean_std(global_aucs),
        "global_f1":  mean_std(global_f1s),
        "global_sen": mean_std(global_sens),
        "equity_auc": mean_std(equities),
        "comm_bytes_per_round": seed_results[0].get("comm_bytes_per_round", 0),
        "n_params": seed_results[0].get("n_params", 0),
    }


if __name__ == "__main__":
    main()
