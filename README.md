# 3D Cell Shape VAE

A **Variational Autoencoder for 3D point cloud shape representation**, demonstrating:

- **PointNet encoder** (permutation-invariant, global max pool)
- **Reparameterisation trick** and ELBO objective
- **Chamfer distance** as reconstruction loss for point clouds
- **Latent space analysis**: UMAP visualisation, latent traversal, geometric descriptor correlation

Developed as a proof-of-concept for 3D biological shape modelling — the same pipeline applies to cell morphology data from fluorescence microscopy.

---

## Motivation

3D cell shape encodes functional state. A VAE that learns a disentangled latent space of shape variation enables:
- Unsupervised cell phenotyping
- Detection of morphological changes under drug treatment or genetic perturbation
- Interpolation between cell states in latent space

This project validates the approach on ModelNet10 (a standard 3D shape benchmark) before applying to biomedical microscopy data.

---

## Results

| Figure | What it shows |
|--------|--------------|
| `fig1_umap_latent.png` | UMAP of latent space — shape categories cluster without supervision |
| `fig2_traversal.png` | Traversing latent dim 0 reveals elongation axis |
| `fig3_descriptor_correlation.png` | Latent dims correlate with geometric descriptors (volume, elongation, symmetry) |

---

## Setup

```bash
# Download ModelNet10
wget http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip -d '/content/project/ModelNet10'
unzip ModelNet10.zip

# Install dependencies
pip install torch numpy matplotlib umap-learn

# Train
python vae_3d.py --data ModelNet10 --epochs 50 --latent 32

# Visualise
python visualise.py --data ModelNet10 --ckpt shape_vae.pt
```

---

## Architecture

```
Input: (B, 1024, 3) point cloud
       │
       ▼
PointNet Encoder
  shared MLP: 3→64→128→256→512
  global max pool → (B, 512)
  fc_mu, fc_logvar → (B, 32)
       │
       ▼ reparameterise: z = mu + eps * std
       │
MLP Decoder
  32→256→512→1024→3072
  reshape → (B, 1024, 3)
       │
       ▼
Loss = Chamfer(recon, input) + β · KL(q(z|x) ‖ p(z))
```

---

## Files

| File | Description |
|------|-------------|
| `vae_3d.py` | Model definition, dataset, training loop, shape descriptors |
| `visualise.py` | UMAP, latent traversal, correlation heatmap |
| `README.md` | This file |

---

## Download the model

- [shape_vae.pt](https://github.com/bhavinmoriya/cell-shape-vae)

---

## Author
Bhavin Moriya, Ph.D. — [github.com/bhavinmoriya](https://github.com/bhavinmoriya)
