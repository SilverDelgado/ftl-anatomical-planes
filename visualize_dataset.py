# Visualizador del dataset del Proyecto 6.
# Uso: uv run python visualize_dataset.py
# Genera: grid de muestras, distribución de clases, comparación de planos anatómicos
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from src.config import HOSPITAL_TRAIN_CSVS, GLOBAL_TEST_CSV, DATA_ROOT, RESULTS_DIR

PLANES  = {1: "Axial", 2: "Coronal", 3: "Sagital", 4: "Mixto"}
CLASSES = {0: "Control", 1: "EM"}
COLORS  = {0: "#4C9BE8", 1: "#E8734C"}

RESULTS_DIR.mkdir(exist_ok=True)


def load_raw_image(img_path: str) -> np.ndarray:
    return np.array(Image.open(DATA_ROOT / img_path).convert("L"))


# ── 1. Muestras de imágenes por hospital y clase ──────────────────────────────

def plot_sample_grid(n_per_class: int = 4):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Muestras del dataset por hospital y clase", fontsize=14, fontweight="bold")

    outer = gridspec.GridSpec(4, 1, hspace=0.45)

    for h_idx, csv in enumerate(HOSPITAL_TRAIN_CSVS):
        df = pd.read_csv(csv)
        inner = gridspec.GridSpecFromSubplotSpec(2, n_per_class, subplot_spec=outer[h_idx], hspace=0.1, wspace=0.05)

        for c_idx, (label_id, label_name) in enumerate(CLASSES.items()):
            subset = df[df["label_id"] == label_id]
            samples = subset.sample(n=min(n_per_class, len(subset)), random_state=42)

            for j, (_, row) in enumerate(samples.iterrows()):
                ax = fig.add_subplot(inner[c_idx, j])
                try:
                    img = load_raw_image(row["image_path"])
                    ax.imshow(img, cmap="gray", aspect="auto")
                except Exception:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                ax.axis("off")
                if j == 0:
                    ax.set_ylabel(label_name, fontsize=8, color=COLORS[label_id], rotation=90, labelpad=5)

        plane_name = df["plane_profile"].iloc[0] if "plane_profile" in df.columns else PLANES[h_idx+1]
        fig.text(
            0.01, outer[h_idx].get_position(fig).y1 - 0.01,
            f"Hospital {h_idx+1} — {PLANES[h_idx+1]} ({plane_name})",
            fontsize=9, fontweight="bold", va="top",
        )

    out = RESULTS_DIR / "muestras_por_hospital.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"Guardado: {out}")


# ── 2. Distribución de clases por hospital ────────────────────────────────────

def plot_class_distribution():
    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    fig.suptitle("Distribución de clases por hospital (train)", fontsize=13, fontweight="bold")

    all_dfs = [pd.read_csv(csv) for csv in HOSPITAL_TRAIN_CSVS]
    combined = pd.concat(all_dfs)

    datasets = [*zip(range(1, 5), all_dfs)] + [(0, combined)]

    for ax, (h_id, df) in zip(axes, datasets):
        counts = df["label_id"].value_counts().sort_index()
        bars = ax.bar(
            [CLASSES[i] for i in counts.index],
            counts.values,
            color=[COLORS[i] for i in counts.index],
            edgecolor="white", linewidth=0.5,
        )
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    str(v), ha="center", va="bottom", fontsize=8)
        total = counts.sum()
        ratio = counts.get(1, 0) / total
        title = f"Hospital {h_id}\n{PLANES[h_id]}" if h_id > 0 else "Global\n(todos)"
        ax.set_title(f"{title}\nn={total}\nEM ratio={ratio:.2f}", fontsize=9)
        ax.set_ylim(0, max(counts.values) * 1.25)
        ax.set_ylabel("Slices" if h_id == 1 else "")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = RESULTS_DIR / "distribucion_clases.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"Guardado: {out}")


# ── 3. Comparación visual de los tres planos para el mismo paciente ───────────

def plot_plane_comparison():
    dfs = {k: pd.read_csv(csv) for k, csv in enumerate(HOSPITAL_TRAIN_CSVS, 1)}

    pats_h1 = set(dfs[1]["patient_id"])
    pats_h2 = set(dfs[2]["patient_id"])
    pats_h3 = set(dfs[3]["patient_id"])
    ms_pats_h1 = set(dfs[1][dfs[1]["label_id"]==1]["patient_id"])

    common = (pats_h1 & pats_h2 & pats_h3 & ms_pats_h1)

    if not common:
        print("No hay pacientes comunes entre H1, H2 y H3 — mostrando pacientes distintos por plano.")
        rows = {
            k: dfs[k][dfs[k]["label_id"]==1].sample(1, random_state=42).iloc[0]
            for k in [1, 2, 3]
        }
    else:
        pat = sorted(common)[0]
        rows = {
            k: dfs[k][dfs[k]["patient_id"]==pat].iloc[len(dfs[k][dfs[k]["patient_id"]==pat])//2]
            for k in [1, 2, 3]
        }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Comparación visual de planos anatómicos (paciente EM)", fontsize=13, fontweight="bold")

    for ax, (k, row) in zip(axes, rows.items()):
        try:
            img = load_raw_image(row["image_path"])
            ax.imshow(img, cmap="gray")
        except Exception as e:
            ax.text(0.5, 0.5, str(e), ha="center", va="center", wrap=True, fontsize=7)
        ax.set_title(f"Hospital {k} — {PLANES[k]}\n{row['patient_id']}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    out = RESULTS_DIR / "comparacion_planos.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"Guardado: {out}")


# ── 4. Resumen estadístico del dataset ────────────────────────────────────────

def print_dataset_summary():
    print("\n" + "="*60)
    print("RESUMEN DEL DATASET — PROYECTO 6")
    print("="*60)

    all_rows = []
    for k, csv in enumerate(HOSPITAL_TRAIN_CSVS, 1):
        df = pd.read_csv(csv)
        counts = df["label_id"].value_counts()
        planes = df["plane"].value_counts() if "plane" in df.columns else {}
        print(f"\nHospital {k} ({PLANES[k]}):")
        print(f"  Total slices : {len(df)}")
        print(f"  Control (0)  : {counts.get(0, 0)}")
        print(f"  EM (1)       : {counts.get(1, 0)}")
        print(f"  EM ratio     : {counts.get(1,0)/len(df):.3f}")
        if len(planes):
            for p, n in planes.items():
                print(f"  Plano {p:10s}: {n}")
        all_rows.append(len(df))

    df_test = pd.read_csv(GLOBAL_TEST_CSV)
    print(f"\nTest global:")
    print(f"  Total slices : {len(df_test)}")
    tc = df_test["label_id"].value_counts()
    print(f"  Control (0)  : {tc.get(0, 0)}")
    print(f"  EM (1)       : {tc.get(1, 0)}")
    print(f"\nTotal train   : {sum(all_rows)}")
    print("="*60)


if __name__ == "__main__":
    print_dataset_summary()
    print("\nGenerando visualizaciones...")
    plot_class_distribution()
    plot_plane_comparison()
    plot_sample_grid(n_per_class=4)
    print("\nListo. Imágenes en results/")
