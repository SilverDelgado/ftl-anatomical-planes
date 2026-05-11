"""Figuras adicionales: per-hospital con AUC, F1, Sen, Spe; equilibrio Sen vs Spe;
Brier por hospital. Para el estudio scratch v2.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent

p = argparse.ArgumentParser()
p.add_argument("--results-dir", type=str, default=str(ROOT / "results_scratch"))
p.add_argument("--out-dir", type=str, default=str(ROOT / "memoria" / "figuras"))
p.add_argument("--suffix", type=str, default="_scratch")
args = p.parse_args()

RES = Path(args.results_dir)
OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)
SUFFIX = args.suffix

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

ALG_COLOR = {"fedavg": "#2E86AB", "fedprox": "#A23B72"}
HOSP_COLOR = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
HOSP_NAME = ["H1 axial", "H2 coronal", "H3 sagital", "H4 mixto"]


def load(name):
    return json.load(open(RES / f"{name}.json"))


def fig_metricas_por_hospital(name):
    """Grid 2x2 de AUC, F1, Sen, Spe por ronda; una curva por hospital."""
    d = load(name)
    s = d["seeds"][0]  # seed 42
    h = s["history"]
    rounds = h["round"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    metricas = [
        ("AUC", "per_client_auc", (0.0, 1.05)),
        ("F1", "per_client_f1", (0.0, 1.05)),
        ("Sensibilidad", "per_client_sen", (-0.02, 1.05)),
        ("Especificidad", "per_client_spe", (-0.02, 1.05)),
    ]
    for ax, (titulo, key, ylim) in zip(axes.flat, metricas):
        per = h.get(key, [])
        if not per:
            ax.set_visible(False)
            continue
        per_T = list(zip(*per))
        for k, vals in enumerate(per_T):
            ax.plot(rounds, vals, color=HOSP_COLOR[k], linewidth=1.6,
                    marker="o", markersize=3.5, label=HOSP_NAME[k])
        ax.set_title(titulo)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
        if titulo == "AUC":
            ax.legend(loc="lower right", fontsize=8, ncol=2)
    for ax in axes[1, :]:
        ax.set_xlabel("ronda")
    fig.suptitle(f"{name} (seed 42): metricas por hospital y ronda", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / f"metricas_por_hospital_{name}{SUFFIX}.pdf")
    plt.close(fig)


def fig_equilibrio_sen_spe():
    """Scatter Sen vs Spe en r5 por algoritmo y hospital. Marca region degenerada."""
    fig, ax = plt.subplots(figsize=(6.2, 5))
    for name in ["fedavg", "fedprox"]:
        d = load(name)
        for s in d["seeds"]:
            for k, c in enumerate(s["final_clients"]):
                if np.isnan(c.get("sensitivity", float("nan"))):
                    continue
                marker = "o" if s["seed"] == 42 else "s"
                ax.scatter(c["sensitivity"], c["specificity"],
                           color=ALG_COLOR[name], marker=marker,
                           edgecolor="black", linewidth=0.4, s=60, alpha=0.85)
    # zonas
    ax.axhspan(0, 0.05, color="red", alpha=0.07, label="region degenerada Spe<=0.05")
    ax.axvspan(0.95, 1.0, color="orange", alpha=0.05)
    # leyenda algoritmos
    handles = []
    for a in ["fedavg", "fedprox"]:
        handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=ALG_COLOR[a],
                                  markeredgecolor="black", markersize=8, label=a))
    handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                              markeredgecolor="black", markersize=8, label="seed 42"))
    handles.append(plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
                              markeredgecolor="black", markersize=8, label="seed 123"))
    ax.legend(handles=handles, loc="lower left", fontsize=8)
    ax.set_xlabel("Sensibilidad (Sen)")
    ax.set_ylabel("Especificidad (Spe)")
    ax.set_title("Equilibrio Sen-Spe en r5 (modelo agregado por hospital)")
    ax.set_xlim(0.85, 1.02); ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / f"equilibrio_sen_spe{SUFFIX}.pdf")
    plt.close(fig)


def fig_brier_por_hospital():
    """Brier por algoritmo y hospital, barras agrupadas."""
    from sklearn.metrics import brier_score_loss
    fig, ax = plt.subplots(figsize=(8, 4))
    metodos = []
    briers_per_h = {}
    for name in ["fedavg", "fedprox"]:
        d = load(name)
        s = d["seeds"][0]  # seed 42
        for k, c in enumerate(s["final_clients"]):
            probs = c.get("probs"); labels = c.get("labels")
            has_nan = False
            if probs is not None:
                for p in probs:
                    if p != p:
                        has_nan = True
                        break
            if probs is None or has_nan:
                br = float("nan")
            else:
                br = brier_score_loss(labels, probs)
            briers_per_h.setdefault(name, []).append(br)
    # tambien centralizado y local
    for ref_name, ref_file, key in [("centralizado", "centralized", "final_clients"),
                                    ("local-only", "local_only", "clients")]:
        d = load(ref_file)
        if key == "final_clients":
            seq = d[key]
        else:
            seq = d[key]
        for c in seq:
            probs = c.get("probs"); labels = c.get("labels")
            if probs is None:
                br = float("nan")
            else:
                br = brier_score_loss(labels, probs)
            briers_per_h.setdefault(ref_name, []).append(br)

    nh = 4
    x = np.arange(nh)
    methods = ["centralizado", "local-only", "fedavg", "fedprox"]
    width = 0.2
    colors = {"centralizado": "#444", "local-only": "#888", "fedavg": ALG_COLOR["fedavg"],
              "fedprox": ALG_COLOR["fedprox"]}
    for i, m in enumerate(methods):
        ax.bar(x + (i - 1.5) * width, briers_per_h.get(m, [0]*nh), width,
               color=colors[m], label=m, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(HOSP_NAME)
    ax.set_ylabel("Brier score (mas bajo = mejor calibracion)")
    ax.set_title("Brier por hospital (seed 42)")
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / f"brier_por_hospital{SUFFIX}.pdf")
    plt.close(fig)


def fig_train_vs_val_per_client():
    """Para FedAvg seed 123 (donde detectamos overfitting moderado): train vs val por cliente."""
    d = load("fedavg")
    # tomar el seed con peor overfitting (123)
    s = None
    for item in d["seeds"]:
        if item["seed"] == 123:
            s = item
            break
    h = s["history"]
    pc_train = h.get("per_client_train_loss_per_epoch", [])
    pc_val = h.get("per_client_val_loss_per_epoch", [])
    if not pc_train:
        return
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    rounds = list(range(1, len(pc_train) + 1))
    for k in range(4):
        ax = axes[k]
        # construir secuencia (round.epoch) en el eje x
        xs, train_seq, val_seq = [], [], []
        for r_idx in range(len(pc_train)):
            for e in range(len(pc_train[r_idx][k])):
                xs.append(r_idx + 1 + (e + 1) / max(len(pc_train[r_idx][k]), 1) * 0.9)
                train_seq.append(pc_train[r_idx][k][e])
                val_seq.append(pc_val[r_idx][k][e])
        ax.plot(xs, train_seq, "s-", color="#E07A1F", linewidth=1.2, markersize=3, label="train_loss")
        ax.plot(xs, val_seq, "o-", color="#2E86AB", linewidth=1.2, markersize=3, label="val_loss")
        ax.set_title(HOSP_NAME[k])
        ax.set_xlabel("ronda.epoca")
        ax.grid(alpha=0.25)
        ax.set_yscale("symlog", linthresh=1e-2)
    axes[0].set_ylabel("BCE")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("FedAvg seed 123: train vs val por cliente y epoca", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / f"train_vs_val_per_client{SUFFIX}.pdf")
    plt.close(fig)


for algo in ["fedavg", "fedprox"]:
    try:
        fig_metricas_por_hospital(algo)
    except Exception as e:
        print(f"  WARN {algo}: {e}")
fig_equilibrio_sen_spe()
fig_brier_por_hospital()
fig_train_vs_val_per_client()

print(f"Figuras adicionales generadas en: {OUT}")
for f in sorted(OUT.glob(f"*{SUFFIX}.pdf")):
    print(f"  {f.name}")
