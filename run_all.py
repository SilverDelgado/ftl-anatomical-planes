import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    HOSPITAL_TRAIN_CSVS, GLOBAL_TEST_CSV, DATA_ROOT, RESULTS_DIR, P6_DIR,
    ROUNDS, LOCAL_EPOCHS, LR, BATCH_SIZE, IMG_SIZE, SEEDS,
    MU_FEDPROX, GRAD_CLIP_NORM,
)
from src.dataset import load_client_datasets, load_test_datasets
from src.baselines import run_local_only, run_centralized, save_results as save
from src.federated import run_federated

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {DEVICE}")

FEDERATED_EXPERIMENTS = {
    "fedavg":  {"algorithm": "fedavg",  "mu": 0.0},
    "fedprox": {"algorithm": "fedprox", "mu": MU_FEDPROX},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["local", "centralized"] + list(FEDERATED_EXPERIMENTS), default=None)
    parser.add_argument("--scratch", action="store_true", help="Entrenar sin pesos preentrenados de ImageNet")
    parser.add_argument("--results-dir", type=str, default=None, help="Sobrescribir el directorio de salida")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--rounds", type=int, default=None)
    args = parser.parse_args()

    pretrained = not args.scratch
    if args.results_dir:
        results_dir = Path(args.results_dir)
    elif args.scratch:
        results_dir = P6_DIR / "results_scratch"
    else:
        results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    lr = args.lr if args.lr is not None else LR
    local_epochs = args.local_epochs if args.local_epochs is not None else LOCAL_EPOCHS
    grad_clip = args.grad_clip if args.grad_clip is not None else GRAD_CLIP_NORM
    weight_decay = args.weight_decay
    print(f"Resultados en: {results_dir}  (pretrained={pretrained})")
    print(f"Hyperparams: lr={lr} epochs={local_epochs} grad_clip={grad_clip} weight_decay={weight_decay}")

    rounds = args.rounds if args.rounds is not None else ROUNDS
    seeds = SEEDS

    print("Cargando datasets...")
    client_train = load_client_datasets(HOSPITAL_TRAIN_CSVS, DATA_ROOT, IMG_SIZE)
    global_test, client_test = load_test_datasets(GLOBAL_TEST_CSV, DATA_ROOT, IMG_SIZE)

    for k, ds in enumerate(client_train, 1):
        print(f"  Hospital {k} train: {len(ds)} slices")
    print(f"  Test global: {len(global_test)} slices")

    run = args.only

    models_dir = results_dir / "models"

    if run in (None, "local"):
        print("\n=== Local-only ===")
        res = run_local_only(
            client_train, global_test, client_test,
            rounds=rounds, local_epochs=local_epochs,
            lr=lr, batch_size=BATCH_SIZE, device=DEVICE, seed=seeds[0],
            weights_dir=models_dir, pretrained=pretrained,
            weight_decay=weight_decay,
        )
        save(res, results_dir / "local_only.json")

    if run in (None, "centralized"):
        print("\n=== Centralizado ===")
        res = run_centralized(
            client_train, global_test, client_test,
            rounds=rounds, local_epochs=local_epochs,
            lr=lr, batch_size=BATCH_SIZE, device=DEVICE, seed=seeds[0],
            weights_save_path=models_dir / "centralized.pt", pretrained=pretrained,
            weight_decay=weight_decay,
        )
        save(res, results_dir / "centralized.json")

    for exp_name, exp_cfg in FEDERATED_EXPERIMENTS.items():
        if run not in (None, exp_name):
            continue
        print(f"\n=== {exp_name} ({exp_cfg}) ===")
        seed_results = []
        for seed in seeds:
            seed_path = results_dir / f"{exp_name}_seed{seed}.json"
            if seed_path.exists():
                import json as _json
                with open(seed_path) as f:
                    res = _json.load(f)
                print(f"\n  Semilla {seed}: ya existe, reanudando desde {seed_path.name}")
                seed_results.append(res)
                continue
            print(f"\n  Semilla {seed}")
            res = run_federated(
                client_train, global_test, client_test,
                rounds=rounds, local_epochs=local_epochs,
                lr=lr, batch_size=BATCH_SIZE, device=DEVICE,
                seed=seed,
                weights_save_path=models_dir / f"{exp_name}_seed{seed}.pt",
                pretrained=pretrained, grad_clip=grad_clip,
                weight_decay=weight_decay,
                **exp_cfg,
            )
            save(res, seed_path)
            seed_results.append(res)
        agg_stats = aggregate_seeds(seed_results)
        save({"seeds": seed_results, "aggregated": agg_stats}, results_dir / f"{exp_name}.json")

    print(f"Todos los experimentos completados. Resultados en {results_dir}/")


def aggregate_seeds(seed_results):
    def mean_std(vals):
        clean = []
        for v in vals:
            if not np.isnan(v):
                clean.append(v)
        return {"mean": float(np.mean(clean)) if clean else float("nan"),
                "std":  float(np.std(clean))  if clean else float("nan")}

    metric_keys = ["auc", "f1", "sensitivity", "specificity", "accuracy"]
    aggregated = {}
    for k in metric_keys:
        metric_vals = []
        for r in seed_results:
            metric_vals.append(r["final_global"].get(k, float("nan")))
        aggregated[f"global_{k}"] = mean_std(metric_vals)

    equity_vals = []
    for r in seed_results:
        equity_vals.append(r["equity_auc"])
    aggregated["equity_auc"] = mean_std(equity_vals)
    aggregated["comm_bytes_per_round"] = seed_results[0].get("comm_bytes_per_round", 0)
    aggregated["n_params"] = seed_results[0].get("n_params", 0)
    return aggregated


if __name__ == "__main__":
    main()
