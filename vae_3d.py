"""
3D Cell Shape VAE — Point Cloud Representation Learning
Demonstrates: VAE, 3D shape modeling, latent space analysis
Dataset: ModelNet10 (public, no registration needed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

# ─────────────────────────────────────────────
# 1. DATA — ModelNet10 point cloud loader
# ─────────────────────────────────────────────

class ModelNet10Dataset(Dataset):
    """
    Loads ModelNet10 .off mesh files, samples N points from surface.
    Download: https://modelnet.cs.princeton.edu/
    Or use the Princeton mirror:
      wget http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
    """
    def __init__(self, root_dir, split='train', n_points=1024, categories=None):
        self.n_points = n_points
        self.files = []
        self.labels = []
        self.label_map = {}

        all_cats = sorted(os.listdir(root_dir))
        all_cats = [c for c in all_cats if os.path.isdir(os.path.join(root_dir, c))]
        if categories:
            all_cats = [c for c in all_cats if c in categories]

        for idx, cat in enumerate(all_cats):
            self.label_map[cat] = idx
            split_dir = os.path.join(root_dir, cat, split)
            if not os.path.exists(split_dir):
                continue
            for f in os.listdir(split_dir):
                if f.endswith('.off'):
                    self.files.append(os.path.join(split_dir, f))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pts = self._load_off(self.files[idx])
        pts = self._sample_points(pts, self.n_points)
        pts = self._normalize(pts)
        return torch.tensor(pts, dtype=torch.float32), self.labels[idx]

    def _load_off(self, path):
        with open(path, 'r', encoding='latin-1') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        # Header: 'OFF' alone or 'OFF12 34 0' on same line
        if lines[0].upper().startswith('OFF'):
            remainder = lines[0][3:].strip()
            if remainder:                        # counts on same line as OFF
                counts_line = remainder
                data_start = 1
            else:                                # counts on next line
                counts_line = lines[1]
                data_start = 2
        else:
            counts_line = lines[0]
            data_start = 1

        n_verts = int(counts_line.split()[0])

        verts = []
        for line in lines[data_start: data_start + n_verts]:
            parts = line.split()
            if len(parts) >= 3:
                verts.append([float(parts[0]), float(parts[1]), float(parts[2])])

        return np.array(verts, dtype=np.float32)

    def _sample_points(self, pts, n):
        if len(pts) >= n:
            idx = np.random.choice(len(pts), n, replace=False)
        else:
            idx = np.random.choice(len(pts), n, replace=True)
        return pts[idx]

    def _normalize(self, pts):
        pts -= pts.mean(axis=0)
        scale = np.max(np.linalg.norm(pts, axis=1))
        return pts / (scale + 1e-8)


# ─────────────────────────────────────────────
# 2. MODEL — PointNet encoder + MLP decoder VAE
# ─────────────────────────────────────────────

class PointNetEncoder(nn.Module):
    """
    PointNet-style encoder: shared MLP → global max pool → latent params.
    Permutation invariant — correct for unordered point clouds.
    """
    def __init__(self, latent_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),   nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 256),nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 512),nn.BatchNorm1d(512), nn.ReLU(),
        )
        self.fc_mu  = nn.Linear(512, latent_dim)
        self.fc_var = nn.Linear(512, latent_dim)

    def forward(self, x):
        # x: (B, N, 3)
        B, N, _ = x.shape
        x_flat = x.reshape(B * N, 3)
        feats = self.mlp(x_flat).reshape(B, N, 512)
        global_feat = feats.max(dim=1).values      # (B, 512) — global max pool
        return self.fc_mu(global_feat), self.fc_var(global_feat)


class MLPDecoder(nn.Module):
    """
    MLP decoder: latent → reconstructed point cloud (B, N*3).
    Folding-style: directly regresses point positions.
    """
    def __init__(self, latent_dim=64, n_points=1024):
        super().__init__()
        self.n_points = n_points
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512),        nn.ReLU(),
            nn.Linear(512, 1024),       nn.ReLU(),
            nn.Linear(1024, n_points * 3),
        )

    def forward(self, z):
        return self.net(z).reshape(z.shape[0], self.n_points, 3)


class ShapeVAE(nn.Module):
    def __init__(self, latent_dim=64, n_points=1024):
        super().__init__()
        self.encoder = PointNetEncoder(latent_dim)
        self.decoder = MLPDecoder(latent_dim, n_points)
        self.latent_dim = latent_dim

    def reparameterise(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterise(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var

    def encode(self, x):
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z):
        return self.decoder(z)


# ─────────────────────────────────────────────
# 3. LOSS — Chamfer distance + KL divergence
# ─────────────────────────────────────────────

def chamfer_distance(pred, target):
    """
    Chamfer distance: mean of nearest-neighbour distances in both directions.
    Differentiable, permutation-invariant reconstruction loss for point clouds.
    pred, target: (B, N, 3)
    """
    diff = pred.unsqueeze(2) - target.unsqueeze(1)   # (B, N, M, 3)
    dist = diff.pow(2).sum(-1)                        # (B, N, M)
    loss = dist.min(dim=2).values.mean() + dist.min(dim=1).values.mean()
    return loss


def vae_loss(recon, x, mu, log_var, beta=1.0):
    recon_loss = chamfer_distance(recon, x)
    kl_loss    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ─────────────────────────────────────────────
# 4. TRAINING LOOP
# ─────────────────────────────────────────────

def train(model, loader, optimizer, device, beta=1.0):
    model.train()
    total, r_total, kl_total = 0, 0, 0
    for pts, _ in loader:
        pts = pts.to(device)
        optimizer.zero_grad()
        recon, mu, log_var = model(pts)
        loss, r_loss, kl = vae_loss(recon, pts, mu, log_var, beta)
        loss.backward()
        optimizer.step()
        total   += loss.item()
        r_total += r_loss.item()
        kl_total+= kl.item()
    n = len(loader)
    return total/n, r_total/n, kl_total/n


# ─────────────────────────────────────────────
# 5. ANALYSIS — Shape descriptors in latent space
# ─────────────────────────────────────────────

def compute_shape_descriptors(pts_batch):
    """
    Compute interpretable geometric descriptors for each point cloud.
    These are the 'ground truth' against which latent dims are correlated.
    """
    descriptors = []
    for pts in pts_batch:
        pts = pts.cpu().numpy()
        # Volume proxy: bounding box volume
        bbox = pts.max(0) - pts.min(0)
        volume = np.prod(bbox + 1e-8)
        # Elongation: ratio of longest to shortest axis
        elongation = bbox.max() / (bbox.min() + 1e-8)
        # Compactness: std of pairwise distances (proxy)
        sample = pts[np.random.choice(len(pts), min(128, len(pts)), replace=False)]
        dists = np.linalg.norm(sample[:, None] - sample[None, :], axis=-1)
        compactness = dists.std()
        # Symmetry: mean distance from point to its mirror across XY plane
        mirrored = pts.copy(); mirrored[:, 2] *= -1
        diff = np.abs(pts[:, None] - mirrored[None, :]).sum(-1)
        symmetry = diff.min(1).mean()
        descriptors.append([volume, elongation, compactness, symmetry])
    return np.array(descriptors)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',     default='ModelNet10', help='Path to ModelNet10 root')
    parser.add_argument('--epochs',   type=int, default=50)
    parser.add_argument('--batch',    type=int, default=32)
    parser.add_argument('--latent',   type=int, default=32)
    parser.add_argument('--npoints',  type=int, default=1024)
    parser.add_argument('--lr',       type=float, default=1e-3)
    parser.add_argument('--beta',     type=float, default=1.0)
    parser.add_argument('--cats',     nargs='+', default=None, help='Subset of categories')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_ds = ModelNet10Dataset(args.data, 'train', args.npoints, args.cats)
    test_ds  = ModelNet10Dataset(args.data, 'test',  args.npoints, args.cats)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=2)

    model = ShapeVAE(latent_dim=args.latent, n_points=args.npoints).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    print(f"Training on {len(train_ds)} shapes | {len(test_ds)} test | latent_dim={args.latent}")

    for epoch in range(1, args.epochs + 1):
        loss, r_loss, kl = train(model, train_loader, optimizer, device, args.beta)
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Loss={loss:.4f} | Recon={r_loss:.4f} | KL={kl:.4f}")

    torch.save(model.state_dict(), 'shape_vae.pt')
    print("Model saved to shape_vae.pt")
    print("Next: run visualise.py to generate latent space figures")
