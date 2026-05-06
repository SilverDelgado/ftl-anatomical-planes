"""Ejecuta todos los experimentos del Proyecto 6 y guarda resultados en results/.

Uso:
    uv run python run_all.py [--dry-run] [--only local|centralized|fedavg|fedprox]

Orden de experimentos:
  1. local_only   — cota inferior (cada hospital por separado)
  2. centralized  — cota superior (todos los datos juntos)
  3. fedavg       — FedAvg con R=50, E=5, K=4
  4. fedprox      — FedProx con mu=0.1, mismos hiperparámetros

Para cada experimento federado se repite con 3 semillas y se promedia.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    HOSPITAL_TRAIN_CSVS, GLOBAL_TEST_CSV, DATA_ROOT, RESULTS_DIR,
    N_CLIENTS, ROUNDS, LOCAL_EPOCHS, LR, BATCH_SIZE, IMG_SIZE,
    SEEDS, MU_FEDPROX,
)
from src.dataset import load_client_datasets, load_test_datasets
from src.baselines import run_local_only, run_centralized, save_results as save
from src.federated import run_federated

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {DEVICE}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Solo 2 rondas, 1 semilla, para verificar")
    parser.add_argument("--only", choices=["local", "centralized", "fedavg", "fedprox"], default=None)
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

    # --- 3. FedAvg ---
    if run in (None, "fedavg"):
        print("\n=== FedAvg ===")
        seed_results = []
        for seed in seeds:
            print(f"\n  Semilla {seed}")
            res = run_federated(
                client_train, global_test, client_test,
                rounds=rounds, local_epochs=LOCAL_EPOCHS,
                lr=LR, batch_size=BATCH_SIZE, device=DEVICE,
                mu=0.0, seed=seed,
            )
            seed_results.append(res)
        agg = _aggregate_seeds(seed_results)
        save({"seeds": seed_results, "aggregated": agg}, RESULTS_DIR / "fedavg.json")

    # --- 4. FedProx ---
    if run in (None, "fedprox"):
        print(f"\n=== FedProx (mu={MU_FEDPROX}) ===")
        seed_results = []
        for seed in seeds:
            print(f"\n  Semilla {seed}")
            res = run_federated(
                client_train, global_test, client_test,
                rounds=rounds, local_epochs=LOCAL_EPOCHS,
                lr=LR, batch_size=BATCH_SIZE, device=DEVICE,
                mu=MU_FEDPROX, seed=seed,
            )
            seed_results.append(res)
        agg = _aggregate_seeds(seed_results)
        save({"seeds": seed_results, "aggregated": agg}, RESULTS_DIR / "fedprox.json")

    print("\nTodos los experimentos completados. Resultados en results/")


def _aggregate_seeds(seed_results: list) -> dict:
    """Promedia métricas finales sobre semillas."""
    def mean_std(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    global_aucs = [r["final_global"]["auc"] for r in seed_results]
    global_f1s  = [r["final_global"]["f1"]  for r in seed_results]
    global_sens = [r["final_global"]["sensitivity"] for r in seed_results]
    equities    = [r["equity_auc"] for r in seed_results]

    return {
        "global_auc":  mean_std(global_aucs),
        "global_f1":   mean_std(global_f1s),
        "global_sen":  mean_std(global_sens),
        "equity_auc":  mean_std(equities),
        "comm_bytes_per_round": seed_results[0]["comm_bytes_per_round"],
    }


if __name__ == "__main__":
    main()
