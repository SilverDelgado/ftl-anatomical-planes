import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", type=str, default=str(ROOT / "results"))
parser.add_argument("--out-dir", type=str, default=str(ROOT / "memoria" / "figuras"))
parser.add_argument("--suffix", type=str, default="", help="Sufijo para los nombres de fichero (ej. _scratch)")
args = parser.parse_args()

RESULTS = Path(args.results_dir)
OUT = Path(args.out_dir)
SUFFIX = args.suffix
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ALG_COLOR = {"fedavg": "#2E86AB", "fedprox": "#A23B72", "central": "#444444"}
HOSP_COLOR = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
HOSP_NAME = ["H1 axial", "H2 coronal", "H3 sagital", "H4 mixto"]


def load(name):
    return json.load(open(RESULTS / f"{name}.json"))


def fig_convergencia():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for name in ["fedavg", "fedprox"]:
        d = load(name)
        for s in d["seeds"]:
            h = s["history"]
            r = h["round"]
            axes[0].plot(r, h["global_auc"], color=ALG_COLOR[name], alpha=0.9 if s["seed"] == 42 else 0.45,
                         linewidth=1.6, label=f"{name} seed {s['seed']}")
            axes[1].plot(r, h["global_val_loss"], color=ALG_COLOR[name], alpha=0.9 if s["seed"] == 42 else 0.45,
                         linewidth=1.6, label=f"{name} seed {s['seed']}")
    dc = load("centralized")
    rc = dc["history"]["round"]
    axes[0].plot(rc, dc["history"]["auc"], color=ALG_COLOR["central"], linestyle="--", linewidth=1.6, label="centralizado")
    axes[1].plot(rc, dc["history"]["val_loss"], color=ALG_COLOR["central"], linestyle="--", linewidth=1.6, label="centralizado")
    axes[0].set_xlabel("ronda"); axes[0].set_ylabel("AUC global"); axes[0].set_title("AUC sobre test global")
    axes[0].set_ylim(0.25, 1.02); axes[0].legend(loc="lower right", ncol=2, fontsize=8)
    axes[1].set_xlabel("ronda"); axes[1].set_ylabel("val\\_loss"); axes[1].set_title("Pérdida sobre test global")
    axes[1].legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"convergencia_global{SUFFIX}.pdf")
    plt.close(fig)


def fig_per_client_auc():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    for ax, name in zip(axes, ["fedavg", "fedprox"]):
        d = load(name)
        s = d["seeds"][0]
        h = s["history"]
        rounds = h["round"]
        per = h["per_client_auc"]
        per_T = list(zip(*per))
        for k, vals in enumerate(per_T):
            ax.plot(rounds, vals, color=HOSP_COLOR[k], linewidth=1.5, label=HOSP_NAME[k])
        ax.set_title(f"{name} (seed 42)")
        ax.set_xlabel("ronda")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("AUC por hospital")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"auc_por_hospital{SUFFIX}.pdf")
    plt.close(fig)


def fig_equity():
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for name in ["fedavg", "fedprox"]:
        d = load(name)
        for s in d["seeds"]:
            h = s["history"]
            ax.plot(h["round"], h["equity_auc"], color=ALG_COLOR[name], alpha=0.9 if s["seed"] == 42 else 0.45,
                    linewidth=1.5, marker="o", markersize=3,
                    label=f"{name} seed {s['seed']}")
    ax.set_xlabel("ronda")
    ax.set_ylabel(r"$\sigma$ AUC entre hospitales (equity)")
    ax.set_title("Inequidad entre hospitales por ronda")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / f"equity{SUFFIX}.pdf")
    plt.close(fig)


def fig_heterogeneidad():
    if not (RESULTS / "heterogeneity.json").exists():
        print("  (skip fig_heterogeneidad: no heterogeneity.json en", RESULTS, ")")
        return
    d = load("heterogeneity")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    cos_mat = np.array(d["gradient_cosine_matrix"])
    im0 = axes[0].imshow(cos_mat, vmin=0.5, vmax=1.0, cmap="viridis")
    axes[0].set_xticks(range(4)); axes[0].set_yticks(range(4))
    axes[0].set_xticklabels(HOSP_NAME, rotation=30, ha="right")
    axes[0].set_yticklabels(HOSP_NAME)
    for i in range(4):
        for j in range(4):
            color = "white" if cos_mat[i, j] < 0.8 else "black"
            axes[0].text(j, i, f"{cos_mat[i,j]:.2f}", ha="center", va="center", color=color, fontsize=10)
    axes[0].set_title(r"Coseno de gradiente entre hospitales")
    fig.colorbar(im0, ax=axes[0], fraction=0.045)

    pairs = ["axial_vs_coronal", "axial_vs_sagittal", "coronal_vs_sagittal"]
    metrics_hist = ["TV", "Hellinger", "Wasserstein"]
    plane_dist = d["plane_distances"]
    width = 0.25
    x = np.arange(len(pairs))
    for i, m in enumerate(metrics_hist):
        vals = []
        for p in pairs:
            vals.append(plane_dist[p][m])
        axes[1].bar(x + (i-1)*width, vals, width, label=m)
    axes[1].set_xticks(x)
    pair_labels = []
    for p in pairs:
        pair_labels.append(p.replace("_", "\n"))
    axes[1].set_xticklabels(pair_labels, fontsize=9)
    axes[1].set_ylabel("distancia (escala mixta)")
    axes[1].set_title("Distancias entre histogramas por plano")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"heterogeneidad{SUFFIX}.pdf")
    plt.close(fig)


def fig_train_vs_val():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    for ax, name in zip(axes, ["fedavg", "fedprox"]):
        d = load(name)
        s = d["seeds"][0]
        h = s["history"]
        rounds_v = h["round"]
        val = h["global_val_loss"]
        # train_loss tiene una entrada por ronda (no incluye r0)
        rounds_t = rounds_v[1:1+len(h["train_loss"])]
        train = h["train_loss"]
        ax.plot(rounds_v, val, "o-", color="#2E86AB", linewidth=1.6, markersize=4, label="val\\_loss (test global)")
        ax.plot(rounds_t, train, "s--", color="#E07A1F", linewidth=1.6, markersize=4, label="train\\_loss (agregada clientes)")
        ax.set_title(f"{name} (seed 42)"); ax.set_xlabel("ronda")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("BCE")
    axes[0].set_yscale("symlog", linthresh=1e-3)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / f"train_vs_val{SUFFIX}.pdf")
    plt.close(fig)


def fig_baseline_summary():
    d_local = load("local_only")
    d_central = load("centralized")
    fa = load("fedavg")["aggregated"]
    fp = load("fedprox")["aggregated"]

    local_f1s = []
    for c in d_local["clients"]:
        local_f1s.append(c["f1"])
    rows = [
        ("Centralizado", d_central["final_global"]["auc"], d_central["final_global"]["f1"], 0.0),
        ("Local-only (mean)", d_local["mean_auc"], np.mean(local_f1s), d_local["equity_auc"]),
        ("FedAvg", fa["global_auc"]["mean"], fa["global_f1"]["mean"], fa["equity_auc"]["mean"]),
        ("FedProx", fp["global_auc"]["mean"], fp["global_f1"]["mean"], fp["equity_auc"]["mean"]),
    ]
    labels = []
    aucs = []
    f1s = []
    eqs = []
    for r in rows:
        labels.append(r[0])
        aucs.append(r[1])
        f1s.append(r[2])
        eqs.append(r[3])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].bar(labels, aucs, color=["#444", "#888", "#2E86AB", "#A23B72"])
    axes[0].set_ylim(0.95, 1.005); axes[0].set_ylabel("AUC global"); axes[0].set_title("AUC final")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(labels, f1s, color=["#444", "#888", "#2E86AB", "#A23B72"])
    axes[1].set_ylim(0.95, 1.005); axes[1].set_ylabel("F1 global"); axes[1].set_title("F1 final")
    axes[1].tick_params(axis="x", rotation=20)
    axes[2].bar(labels, eqs, color=["#444", "#888", "#2E86AB", "#A23B72"])
    axes[2].set_ylabel(r"$\sigma$ AUC entre hospitales"); axes[2].set_title("Equity (más bajo = mejor)")
    axes[2].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUT / f"resumen_metricas{SUFFIX}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_convergencia()
    fig_per_client_auc()
    fig_equity()
    fig_heterogeneidad()
    fig_train_vs_val()
    fig_baseline_summary()
    print(f"Figuras generadas en: {OUT}")
    for f in sorted(OUT.glob("*.pdf")):
        print(f"  {f.name}")
