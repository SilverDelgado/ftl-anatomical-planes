import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as T
from scipy.linalg import sqrtm

P6 = Path(__file__).parent.parent
CSVS = P6.parent / "federated_csvs" / "06_hfl_cls_plane_comparison_flair"
RES = P6 / "results"
RES.mkdir(exist_ok=True)

NBINS = 64
SAMPLES_PER_PLANE = 800
SAMPLES_PER_HOSPITAL = 400
EMB_BATCH = 32
DEVICE = torch.device("cpu")

_IMG_MEAN = [0.485, 0.456, 0.406]
_IMG_STD = [0.229, 0.224, 0.225]


def img_hist(path, nbins=NBINS):
    arr = np.array(Image.open(path).convert("L"), dtype=np.uint8).ravel()
    hist, _ = np.histogram(arr, bins=nbins, range=(0, 256), density=False)
    return hist


def average_hist(image_paths, nbins=NBINS):
    h = np.zeros(nbins, dtype=np.float64)
    for p in image_paths:
        h += img_hist(P6 / p, nbins)
    h = h / max(h.sum(), 1.0)
    return h


def total_variation(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def hellinger(p, q):
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def kl_div(p, q, eps=1e-12):
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return float(np.sum(p * np.log(p / q)))


class ImagePathDataset(Dataset):
    def __init__(self, paths, img_size=224):
        self.paths = list(paths)
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
            T.Normalize(_IMG_MEAN, _IMG_STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(P6 / self.paths[idx]).convert("L")
        return self.transform(img)


def build_encoder():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.eval()
    return model.to(DEVICE)


@torch.no_grad()
def extract_embeddings(encoder, paths):
    ds = ImagePathDataset(paths)
    loader = DataLoader(ds, batch_size=EMB_BATCH, shuffle=False, num_workers=0)
    feats = []
    for batch in loader:
        f = encoder(batch.to(DEVICE)).cpu().numpy()
        feats.append(f)
    return np.concatenate(feats, axis=0)


def frechet_distance(mu_a, sigma_a, mu_b, sigma_b, eps=1e-6):
    diff = mu_a - mu_b
    covmean = sqrtm(sigma_a @ sigma_b)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_a.shape[0]) * eps
        covmean = sqrtm((sigma_a + offset) @ (sigma_b + offset))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_a + sigma_b - 2 * covmean))


def cosine_similarity_means(mu_a, mu_b):
    return float(np.dot(mu_a, mu_b) / (np.linalg.norm(mu_a) * np.linalg.norm(mu_b) + 1e-12))


def gaussian_stats(features):
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def main():
    print("Cargando muestras por plano...")
    train_dfs = [pd.read_csv(CSVS / f"hospital_{k}_train.csv") for k in range(1, 5)]
    full = pd.concat(train_dfs, ignore_index=True)

    plane_samples = {}
    plane_hists = {}
    for plane in ("axial", "coronal", "sagittal"):
        sub = full[full.plane == plane]
        sample = sub.sample(min(SAMPLES_PER_PLANE, len(sub)), random_state=0)
        plane_samples[plane] = list(sample.image_path)
        plane_hists[plane] = average_hist(plane_samples[plane])
        print(f"  {plane}: {len(sample)} slices muestreadas")

    print()
    print("=== Distancias entre planos (histograma de pixel) ===")
    plane_pairs = [("axial", "coronal"), ("axial", "sagittal"), ("coronal", "sagittal")]
    plane_hist_dists = {}
    for a, b in plane_pairs:
        tv = total_variation(plane_hists[a], plane_hists[b])
        hh = hellinger(plane_hists[a], plane_hists[b])
        klp = kl_div(plane_hists[a], plane_hists[b])
        plane_hist_dists[f"{a}_vs_{b}"] = {"TV": tv, "Hellinger": hh, "KL": klp}
        print(f"  {a:8s} vs {b:8s}  TV={tv:.4f}  Hellinger={hh:.4f}  KL={klp:.4f}")

    print()
    print("Cargando muestras por hospital...")
    hospital_samples = {}
    hospital_hists = {}
    for k in range(1, 5):
        df = train_dfs[k - 1]
        sample = df.sample(min(SAMPLES_PER_HOSPITAL, len(df)), random_state=0)
        hospital_samples[k] = list(sample.image_path)
        hospital_hists[k] = average_hist(hospital_samples[k])
        print(f"  H{k} ({df.plane_profile.iloc[0]}): {len(sample)} slices muestreadas")

    print()
    print("=== Distancias entre hospitales (histograma de pixel) ===")
    print(f"  {'par':12s} {'TV':>7s} {'Hellinger':>10s} {'KL':>8s}")
    hospital_hist_dists = {}
    for i in range(1, 5):
        for j in range(i + 1, 5):
            tv = total_variation(hospital_hists[i], hospital_hists[j])
            hh = hellinger(hospital_hists[i], hospital_hists[j])
            klp = kl_div(hospital_hists[i], hospital_hists[j])
            hospital_hist_dists[f"H{i}_vs_H{j}"] = {"TV": tv, "Hellinger": hh, "KL": klp}
            print(f"  H{i}_vs_H{j:<6d} {tv:7.4f} {hh:10.4f} {klp:8.4f}")

    print()
    print(f"Extrayendo embeddings con ResNet-18 (ImageNet) en {DEVICE}...")
    encoder = build_encoder()
    plane_emb = {}
    plane_stats = {}
    for plane, paths in plane_samples.items():
        feats = extract_embeddings(encoder, paths)
        plane_emb[plane] = feats
        plane_stats[plane] = gaussian_stats(feats)
        print(f"  {plane}: features {feats.shape}")
    hospital_emb = {}
    hospital_stats = {}
    for k, paths in hospital_samples.items():
        feats = extract_embeddings(encoder, paths)
        hospital_emb[k] = feats
        hospital_stats[k] = gaussian_stats(feats)
        print(f"  H{k}: features {feats.shape}")

    print()
    print("=== Distancias entre planos (embeddings ResNet-18, 512-d) ===")
    print(f"  {'par':24s} {'||mu_a-mu_b||':>14s} {'Frechet':>10s} {'cos(mu_a,mu_b)':>15s}")
    plane_emb_dists = {}
    for a, b in plane_pairs:
        mu_a, sig_a = plane_stats[a]
        mu_b, sig_b = plane_stats[b]
        dmean = float(np.linalg.norm(mu_a - mu_b))
        fd = frechet_distance(mu_a, sig_a, mu_b, sig_b)
        cs = cosine_similarity_means(mu_a, mu_b)
        plane_emb_dists[f"{a}_vs_{b}"] = {"mean_dist": dmean, "frechet": fd, "cosine": cs}
        print(f"  {a:8s} vs {b:8s}        {dmean:14.3f} {fd:10.3f} {cs:15.4f}")

    print()
    print("=== Distancias entre hospitales (embeddings) ===")
    print(f"  {'par':12s} {'||mu_a-mu_b||':>14s} {'Frechet':>10s} {'cos(mu_a,mu_b)':>15s}")
    hospital_emb_dists = {}
    for i in range(1, 5):
        for j in range(i + 1, 5):
            mu_a, sig_a = hospital_stats[i]
            mu_b, sig_b = hospital_stats[j]
            dmean = float(np.linalg.norm(mu_a - mu_b))
            fd = frechet_distance(mu_a, sig_a, mu_b, sig_b)
            cs = cosine_similarity_means(mu_a, mu_b)
            hospital_emb_dists[f"H{i}_vs_H{j}"] = {"mean_dist": dmean, "frechet": fd, "cosine": cs}
            print(f"  H{i}_vs_H{j:<6d} {dmean:14.3f} {fd:10.3f} {cs:15.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    bins = np.linspace(0, 256, NBINS + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    for plane, h in plane_hists.items():
        axes[0][0].plot(centers, h, label=plane, lw=2)
    axes[0][0].set_title("Histograma de intensidad por plano")
    axes[0][0].set_xlabel("intensidad de pixel"); axes[0][0].set_ylabel("densidad")
    axes[0][0].legend(); axes[0][0].grid(alpha=0.3)

    for k, h in hospital_hists.items():
        axes[0][1].plot(centers, h, label=f"H{k} ({train_dfs[k-1].plane_profile.iloc[0]})", lw=2)
    axes[0][1].set_title("Histograma por hospital")
    axes[0][1].set_xlabel("intensidad de pixel"); axes[0][1].set_ylabel("densidad")
    axes[0][1].legend(); axes[0][1].grid(alpha=0.3)

    plane_labels = list(plane_pairs)
    plane_fds = [plane_emb_dists[f"{a}_vs_{b}"]["frechet"] for a, b in plane_labels]
    plane_dms = [plane_emb_dists[f"{a}_vs_{b}"]["mean_dist"] for a, b in plane_labels]
    x = np.arange(len(plane_labels))
    axes[1][0].bar(x - 0.18, plane_dms, 0.36, label="||mu_a - mu_b||")
    ax2 = axes[1][0].twinx()
    ax2.bar(x + 0.18, plane_fds, 0.36, color="tab:orange", label="Frechet")
    axes[1][0].set_xticks(x)
    axes[1][0].set_xticklabels([f"{a}_vs_{b}" for a, b in plane_labels])
    axes[1][0].set_title("Distancia entre planos en el espacio de embeddings")
    axes[1][0].set_ylabel("||mu_a - mu_b||")
    ax2.set_ylabel("Frechet distance")
    axes[1][0].grid(axis="y", alpha=0.3)
    axes[1][0].legend(loc="upper left"); ax2.legend(loc="upper right")

    pairs = [(i, j) for i in range(1, 5) for j in range(i + 1, 5)]
    h_fds = [hospital_emb_dists[f"H{i}_vs_H{j}"]["frechet"] for i, j in pairs]
    h_dms = [hospital_emb_dists[f"H{i}_vs_H{j}"]["mean_dist"] for i, j in pairs]
    x = np.arange(len(pairs))
    axes[1][1].bar(x - 0.18, h_dms, 0.36, label="||mu_a - mu_b||")
    ax2 = axes[1][1].twinx()
    ax2.bar(x + 0.18, h_fds, 0.36, color="tab:orange", label="Frechet")
    axes[1][1].set_xticks(x)
    axes[1][1].set_xticklabels([f"H{i}-H{j}" for i, j in pairs], rotation=20)
    axes[1][1].set_title("Distancia entre hospitales en embeddings")
    axes[1][1].set_ylabel("||mu_a - mu_b||")
    ax2.set_ylabel("Frechet distance")
    axes[1][1].grid(axis="y", alpha=0.3)
    axes[1][1].legend(loc="upper left"); ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(RES / "heterogeneity_hist.png", dpi=150)
    print()
    print(f"Figura guardada en {RES / 'heterogeneity_hist.png'}")

    out = {
        "config": {
            "nbins": NBINS,
            "samples_per_plane": SAMPLES_PER_PLANE,
            "samples_per_hospital": SAMPLES_PER_HOSPITAL,
            "embedding_dim": int(plane_emb["axial"].shape[1]),
        },
        "plane_distances": plane_hist_dists,
        "hospital_distances": hospital_hist_dists,
        "plane_distances_embeddings": plane_emb_dists,
        "hospital_distances_embeddings": hospital_emb_dists,
        "plane_histograms": {p: h.tolist() for p, h in plane_hists.items()},
        "hospital_histograms": {f"H{k}": h.tolist() for k, h in hospital_hists.items()},
    }
    with open(RES / "heterogeneity.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"JSON guardado en {RES / 'heterogeneity.json'}")


if __name__ == "__main__":
    main()
