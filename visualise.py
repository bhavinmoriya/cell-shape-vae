"""
Latent Space Visualisation for ShapeVAE
Produces 3 publication-ready figures:
  1. UMAP of latent space coloured by shape category
  2. Latent traversal — vary one dim, reconstruct shapes
  3. Correlation heatmap: latent dims vs. geometric descriptors
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader

# ── lazy imports (install if absent) ──────────────────────
try:
    import umap
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'umap-learn', '-q'])
    import umap

from vae_3d import ShapeVAE, ModelNet10Dataset, compute_shape_descriptors

COLORS = ['#4e79a7','#f28e2b','#e15759','#76b7b2',
          '#59a14f','#edc948','#b07aa1','#ff9da7',
          '#9c755f','#bab0ac']


def encode_dataset(model, loader, device):
    model.eval()
    zs, labels, pts_all = [], [], []
    with torch.no_grad():
        for pts, lbl in loader:
            z = model.encode(pts.to(device))
            zs.append(z.cpu().numpy())
            labels.extend(lbl.numpy())
            pts_all.append(pts)
    return np.vstack(zs), np.array(labels), torch.cat(pts_all)


def fig1_umap(zs, labels, label_names, save='fig1_umap_latent.png'):
    """UMAP projection of latent space coloured by shape class."""
    print("Computing UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    emb = reducer.fit_transform(zs)

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, name in enumerate(label_names):
        mask = labels == i
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=COLORS[i % len(COLORS)], label=name,
                   alpha=0.7, s=18, linewidths=0)
    ax.set_title("VAE Latent Space — UMAP Projection\n(ModelNet10 shape categories)",
                 fontsize=13, pad=12)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=2, framealpha=0.9, fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(save, dpi=150)
    plt.close()
    print(f"Saved {save}")


def fig2_latent_traversal(model, zs, device, dim=0, n_steps=7, save='fig2_traversal.png'):
    """
    Fix all latent dims at their mean, traverse one dim across ±3 std.
    Visualises what that dimension encodes about shape.
    """
    model.eval()
    z_mean = torch.tensor(zs.mean(0), dtype=torch.float32)
    z_std  = zs.std(0)[dim]
    values = np.linspace(-3 * z_std, 3 * z_std, n_steps)

    fig = plt.figure(figsize=(n_steps * 2.2, 3))
    fig.suptitle(f"Latent Traversal — Dimension {dim}  (all other dims fixed at mean)",
                 fontsize=11, y=1.02)

    for i, v in enumerate(values):
        z = z_mean.clone()
        z[dim] = v
        with torch.no_grad():
            pts = model.decode(z.unsqueeze(0).to(device))[0].cpu().numpy()

        ax = fig.add_subplot(1, n_steps, i + 1, projection='3d')
        ax.scatter(pts[::4, 0], pts[::4, 1], pts[::4, 2],
                   s=2, c='#4e79a7', alpha=0.6)
        ax.set_title(f"{v:+.1f}σ", fontsize=8)
        ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save}")


def fig3_descriptor_correlation(zs, pts_all, save='fig3_descriptor_correlation.png'):
    """
    Pearson correlation between latent dims and geometric shape descriptors.
    Shows the VAE has learned interpretable structure.
    """
    print("Computing shape descriptors...")
    descs = compute_shape_descriptors(pts_all[:min(500, len(pts_all))])
    z_sub = zs[:len(descs)]

    desc_names = ['Volume', 'Elongation', 'Compactness', 'Symmetry']
    # Top 10 latent dims by total absolute correlation
    corr_matrix = np.corrcoef(z_sub.T, descs.T)[:z_sub.shape[1], z_sub.shape[1]:]
    top_dims = np.argsort(-np.abs(corr_matrix).sum(1))[:10]
    corr_top = corr_matrix[top_dims]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_top.T, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(top_dims)))
    ax.set_xticklabels([f"z{d}" for d in top_dims], fontsize=9)
    ax.set_yticks(range(4))
    ax.set_yticklabels(desc_names, fontsize=10)
    ax.set_title("Latent Dim vs. Geometric Descriptor Correlation\n(top 10 dims by |ρ|)",
                 fontsize=11)
    fig.colorbar(im, ax=ax, label="Pearson ρ")
    plt.tight_layout()
    plt.savefig(save, dpi=150)
    plt.close()
    print(f"Saved {save}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',    default='ModelNet10')
    parser.add_argument('--ckpt',    default='shape_vae.pt')
    parser.add_argument('--latent',  type=int, default=32)
    parser.add_argument('--npoints', type=int, default=1024)
    parser.add_argument('--batch',   type=int, default=32)
    parser.add_argument('--cats',    nargs='+', default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ShapeVAE(latent_dim=args.latent, n_points=args.npoints).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    ds     = ModelNet10Dataset(args.data, 'test', args.npoints, args.cats)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2)
    label_names = [k for k, v in sorted(ds.label_map.items(), key=lambda x: x[1])]

    zs, labels, pts_all = encode_dataset(model, loader, device)
    print(f"Encoded {len(zs)} shapes | latent shape: {zs.shape}")

    fig1_umap(zs, labels, label_names)
    fig2_latent_traversal(model, zs, device, dim=0)
    fig3_descriptor_correlation(zs, pts_all)
    print("\nAll figures saved. Upload to GitHub with README.")
