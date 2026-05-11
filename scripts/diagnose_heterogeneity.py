import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import torchvision.transforms as T
from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance

P6 = Path(__file__).parent.parent
CSVS = P6.parent / "federated_csvs" / "06_hfl_cls_plane_comparison_flair"
RES = P6 / "results"
RES.mkdir(exist_ok=True)

NBINS = 64
SAMPLES_PER_PLANE = 800
SAMPLES_PER_HOSPITAL = 400
EMB_BATCH = 32
GRAD_BATCH = 32
GRAD_N_BATCHES = 30
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


def wasserstein_1d(p, q, bin_centers):
    return float(wasserstein_distance(bin_centers, bin_centers, u_weights=p, v_weights=q))


class ImagePathDataset(Dataset):
    def __init__(self, paths, labels=None, img_size=224):
        self.paths = list(paths)
        self.labels = list(labels) if labels is not None else None
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
        x = self.transform(img)
        if self.labels is None:
            return x
        return x, torch.tensor(float(self.labels[idx]), dtype=torch.float32)


def build_encoder():
    model = resnet18(weights=None)
    model.fc = nn.Identity()
    model.eval()
    return model.to(DEVICE)


def build_classifier():
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, 1)
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


def average_gradient(model, paths, labels, n_batches=GRAD_N_BATCHES, batch_size=GRAD_BATCH, seed=0):
    torch.manual_seed(seed)
    ds = ImagePathDataset(paths, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    accumulated = None
    n_seen = 0
    for batch_idx, (X, y) in enumerate(loader):
        if batch_idx >= n_batches:
            break
        X = X.to(DEVICE); y = y.to(DEVICE).unsqueeze(1)
        model.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.flatten())
        flat = torch.cat(grads)
        if accumulated is None:
            accumulated = flat.detach().clone()
        else:
            accumulated += flat.detach()
        n_seen += 1
    return (accumulated / max(n_seen, 1)).cpu().numpy()


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
    train_dfs = []
    for k in range(1, 5):
        train_dfs.append(pd.read_csv(CSVS / f"hospital_{k}_train.csv"))
    full = pd.concat(train_dfs, ignore_index=True)

    plane_samples = {}
    plane_hists = {}
    for plane in ("axial", "coronal", "sagittal"):
        sub = full[full.plane == plane]
        sample = sub.sample(min(SAMPLES_PER_PLANE, len(sub)), random_state=0)
        plane_samples[plane] = list(sample.image_path)
        plane_hists[plane] = average_hist(plane_samples[plane])
        print(f"  {plane}: {len(sample)} slices muestreadas")

    bins = np.linspace(0, 256, NBINS + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    print()
    print("=== Distancias entre planos (histograma de pixel) ===")
    plane_pairs = [("axial", "coronal"), ("axial", "sagittal"), ("coronal", "sagittal")]
    plane_hist_dists = {}
    for a, b in plane_pairs:
        tv = total_variation(plane_hists[a], plane_hists[b])
        hh = hellinger(plane_hists[a], plane_hists[b])
        klp = kl_div(plane_hists[a], plane_hists[b])
        ws = wasserstein_1d(plane_hists[a], plane_hists[b], bin_centers)
        plane_hist_dists[f"{a}_vs_{b}"] = {"TV": tv, "Hellinger": hh, "KL": klp, "Wasserstein": ws}
        print(f"  {a:8s} vs {b:8s}  TV={tv:.4f}  Hellinger={hh:.4f}  KL={klp:.4f}  W1={ws:.3f}")

    print()
    print("Cargando muestras por hospital...")
    hospital_samples = {}
    hospital_labels = {}
    hospital_hists = {}
    for k in range(1, 5):
        df = train_dfs[k - 1]
        sample = df.sample(min(SAMPLES_PER_HOSPITAL, len(df)), random_state=0)
        hospital_samples[k] = list(sample.image_path)
        hospital_labels[k] = list(sample.label_id)
        hospital_hists[k] = average_hist(hospital_samples[k])
        print(f"  H{k} ({df.plane_profile.iloc[0]}): {len(sample)} slices muestreadas")

    print()
    print("=== Distancias entre hospitales (histograma de pixel) ===")
    print(f"  {'par':12s} {'TV':>7s} {'Hellinger':>10s} {'KL':>8s} {'W1':>8s}")
    hospital_hist_dists = {}
    for i in range(1, 5):
        for j in range(i + 1, 5):
            tv = total_variation(hospital_hists[i], hospital_hists[j])
            hh = hellinger(hospital_hists[i], hospital_hists[j])
            klp = kl_div(hospital_hists[i], hospital_hists[j])
            ws = wasserstein_1d(hospital_hists[i], hospital_hists[j], bin_centers)
            hospital_hist_dists[f"H{i}_vs_H{j}"] = {"TV": tv, "Hellinger": hh, "KL": klp, "Wasserstein": ws}
            print(f"  H{i}_vs_H{j:<6d} {tv:7.4f} {hh:10.4f} {klp:8.4f} {ws:8.3f}")

    print()
    print(f"Extrayendo embeddings con ResNet-18 sin entrenar en {DEVICE}...")
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
    plane_emb_dists = {}
    for a, b in plane_pairs:
        mu_a, sig_a = plane_stats[a]
        mu_b, sig_b = plane_stats[b]
        dmean = float(np.linalg.norm(mu_a - mu_b))
        fd = frechet_distance(mu_a, sig_a, mu_b, sig_b)
        cs = cosine_similarity_means(mu_a, mu_b)
        plane_emb_dists[f"{a}_vs_{b}"] = {"mean_dist": dmean, "frechet": fd, "cosine": cs}
        print(f"  {a:8s} vs {b:8s}  ||mu||={dmean:.3f}  Frechet={fd:.3f}  cos={cs:.4f}")

    print()
    print("=== Distancias entre hospitales (embeddings) ===")
    hospital_emb_dists = {}
    for i in range(1, 5):
        for j in range(i + 1, 5):
            mu_a, sig_a = hospital_stats[i]
            mu_b, sig_b = hospital_stats[j]
            dmean = float(np.linalg.norm(mu_a - mu_b))
            fd = frechet_distance(mu_a, sig_a, mu_b, sig_b)
            cs = cosine_similarity_means(mu_a, mu_b)
            hospital_emb_dists[f"H{i}_vs_H{j}"] = {"mean_dist": dmean, "frechet": fd, "cosine": cs}
            print(f"  H{i}_vs_H{j:<6d}  ||mu||={dmean:.3f}  Frechet={fd:.3f}  cos={cs:.4f}")

    print()
    print(f"Calculando gradientes por hospital con ResNet-18 sin entrenar (BCE) en {DEVICE}...")
    print(f"  {GRAD_N_BATCHES} batches x {GRAD_BATCH} imgs por hospital. Esto tarda algunos minutos en CPU.")
    classifier = build_classifier()
    init_state = {k: v.detach().clone() for k, v in classifier.state_dict().items()}

    hospital_grads = {}
    for k in range(1, 5):
        classifier.load_state_dict(init_state)
        g = average_gradient(classifier, hospital_samples[k], hospital_labels[k], seed=k)
        hospital_grads[k] = g
        print(f"  H{k}: gradiente shape {g.shape}, ||g||={np.linalg.norm(g):.3f}")

    print()
    print("=== Métricas basadas en gradientes (T6.6.5), ResNet-18 sin entrenar ===")
    print(f"  {'par':12s} {'cos(g_i, g_j)':>15s} {'||g_i - g_j||':>15s}")
    hospital_grad_dists = {}
    cos_matrix = np.eye(4)
    div_matrix = np.zeros((4, 4))
    for i in range(1, 5):
        for j in range(1, 5):
            if i == j:
                continue
            gi, gj = hospital_grads[i], hospital_grads[j]
            cos_ij = float(np.dot(gi, gj) / (np.linalg.norm(gi) * np.linalg.norm(gj) + 1e-12))
            div_ij = float(np.linalg.norm(gi - gj))
            cos_matrix[i - 1, j - 1] = cos_ij
            div_matrix[i - 1, j - 1] = div_ij
            if i < j:
                hospital_grad_dists[f"H{i}_vs_H{j}"] = {"cosine": cos_ij, "divergence": div_ij}
                print(f"  H{i}_vs_H{j:<6d}  {cos_ij:15.4f}  {div_ij:15.3f}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for plane, h in plane_hists.items():
        axes[0][0].plot(bin_centers, h, label=plane, lw=2)
    axes[0][0].set_title("Histograma de intensidad por plano")
    axes[0][0].set_xlabel("intensidad de pixel"); axes[0][0].set_ylabel("densidad")
    axes[0][0].legend(); axes[0][0].grid(alpha=0.3)

    for k, h in hospital_hists.items():
        axes[0][1].plot(bin_centers, h, label=f"H{k} ({train_dfs[k-1].plane_profile.iloc[0]})", lw=2)
    axes[0][1].set_title("Histograma por hospital")
    axes[0][1].set_xlabel("intensidad de pixel"); axes[0][1].set_ylabel("densidad")
    axes[0][1].legend(); axes[0][1].grid(alpha=0.3)

    pairs = []
    for i in range(1, 5):
        for j in range(i + 1, 5):
            pairs.append((i, j))
    h_fds = []
    h_dms = []
    pair_labels = []
    for i, j in pairs:
        h_fds.append(hospital_emb_dists[f"H{i}_vs_H{j}"]["frechet"])
        h_dms.append(hospital_emb_dists[f"H{i}_vs_H{j}"]["mean_dist"])
        pair_labels.append(f"H{i}-H{j}")
    x = np.arange(len(pairs))
    axes[0][2].bar(x - 0.18, h_dms, 0.36, label="||mu_a - mu_b||")
    ax2 = axes[0][2].twinx()
    ax2.bar(x + 0.18, h_fds, 0.36, color="tab:orange", label="Frechet")
    axes[0][2].set_xticks(x)
    axes[0][2].set_xticklabels(pair_labels, rotation=20)
    axes[0][2].set_title("Distancia entre hospitales (embeddings)")
    axes[0][2].set_ylabel("||mu_a - mu_b||"); ax2.set_ylabel("Frechet distance")
    axes[0][2].grid(axis="y", alpha=0.3)
    axes[0][2].legend(loc="upper left"); ax2.legend(loc="upper right")

    plane_w = []
    plane_tv = []
    plane_labels = []
    for a, b in plane_pairs:
        plane_w.append(plane_hist_dists[f"{a}_vs_{b}"]["Wasserstein"])
        plane_tv.append(plane_hist_dists[f"{a}_vs_{b}"]["TV"])
        plane_labels.append(f"{a[:3]}-{b[:3]}")
    x = np.arange(len(plane_pairs))
    axes[1][0].bar(x - 0.18, plane_tv, 0.36, label="TV")
    ax3 = axes[1][0].twinx()
    ax3.bar(x + 0.18, plane_w, 0.36, color="tab:purple", label="Wasserstein-1")
    axes[1][0].set_xticks(x)
    axes[1][0].set_xticklabels(plane_labels)
    axes[1][0].set_title("TV vs Wasserstein entre planos")
    axes[1][0].set_ylabel("TV"); ax3.set_ylabel("W1 (intensidad)")
    axes[1][0].grid(axis="y", alpha=0.3)
    axes[1][0].legend(loc="upper left"); ax3.legend(loc="upper right")

    im1 = axes[1][1].imshow(cos_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[1][1].set_xticks(range(4)); axes[1][1].set_yticks(range(4))
    axes[1][1].set_xticklabels(["H1", "H2", "H3", "H4"])
    axes[1][1].set_yticklabels(["H1", "H2", "H3", "H4"])
    axes[1][1].set_title("cos(grad_i, grad_j), ResNet-18 sin entrenar")
    for i in range(4):
        for j in range(4):
            axes[1][1].text(j, i, f"{cos_matrix[i, j]:.2f}", ha="center", va="center")
    plt.colorbar(im1, ax=axes[1][1], fraction=0.046)

    im2 = axes[1][2].imshow(div_matrix, cmap="viridis")
    axes[1][2].set_xticks(range(4)); axes[1][2].set_yticks(range(4))
    axes[1][2].set_xticklabels(["H1", "H2", "H3", "H4"])
    axes[1][2].set_yticklabels(["H1", "H2", "H3", "H4"])
    axes[1][2].set_title("||grad_i - grad_j||, ResNet-18 sin entrenar")
    for i in range(4):
        for j in range(4):
            axes[1][2].text(j, i, f"{div_matrix[i, j]:.1f}", ha="center", va="center", color="white")
    plt.colorbar(im2, ax=axes[1][2], fraction=0.046)

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
            "grad_batches": GRAD_N_BATCHES,
            "grad_batch_size": GRAD_BATCH,
        },
        "plane_distances": plane_hist_dists,
        "hospital_distances": hospital_hist_dists,
        "plane_distances_embeddings": plane_emb_dists,
        "hospital_distances_embeddings": hospital_emb_dists,
        "hospital_distances_gradients": hospital_grad_dists,
        "gradient_cosine_matrix": cos_matrix.tolist(),
        "gradient_divergence_matrix": div_matrix.tolist(),
        "plane_histograms": {p: h.tolist() for p, h in plane_hists.items()},
        "hospital_histograms": {f"H{k}": h.tolist() for k, h in hospital_hists.items()},
    }
    with open(RES / "heterogeneity.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"JSON guardado en {RES / 'heterogeneity.json'}")


if __name__ == "__main__":
    main()
