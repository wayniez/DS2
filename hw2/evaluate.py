"""
Evaluation and visualisation of anomalies.

Run:
    python evaluate.py --category pcb1 --checkpoint checkpoints/pcb1_best.pth
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from dataset import build_dataloaders
from model import SiameseAutoencoder


def anomaly_score(x, x_hat):
    """Per-pixel reconstruction error → scalar score."""
    err = (x - x_hat) ** 2              # [B, 3, H, W]
    heatmap = err.mean(dim=1)            # [B, H, W] — average across channels
    k = max(1, int(heatmap.shape[-1] * heatmap.shape[-2] * 0.1))
    score = heatmap.flatten(1).topk(k, dim=1).values.mean(dim=1)
    return score, heatmap


def denormalize(t):
    """Return the tensor to the range [0, 1] for display."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t.cpu() * std + mean).clamp(0, 1)


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, test_loader = build_dataloaders(
        root=args.data_root,
        category=args.category,
        img_size=args.img_size,
        batch_size=1,
    )

    model = SiameseAutoencoder(latent_dim=args.latent_dim).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    scores = []
    labels = []

    heatmaps = []
    images = []
    recons = []

    pixel_preds = []
    pixel_targets = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["image"].to(device)
            x_hat = model.reconstruct(x)

            score, hmap = anomaly_score(x, x_hat)

            scores.append(score.item())
            labels.append(batch["label"].item())

            heatmaps.append(hmap.squeeze().cpu().numpy())
            images.append(denormalize(x.squeeze()))
            recons.append(denormalize(x_hat.squeeze()))

            mask = batch["mask"]

            if batch["has_mask"][0]:
                pixel_preds.extend(hmap.flatten().cpu().numpy())
                pixel_targets.extend((mask.flatten().cpu().numpy() > 0.5).astype(int))

    scores = np.array(scores)
    labels = np.array(labels)

    # Image-level ROC-AUC
    auc = roc_auc_score(labels, scores)
    print(f"\nROC-AUC [{args.category}]: {auc:.4f}")

    if len(pixel_targets) > 0:
        pixel_targets_arr = np.array(pixel_targets)
        pixel_preds_arr = np.array(pixel_preds)

        if len(np.unique(pixel_targets_arr)) < 2:
            print(
                f"Pixel ROC-AUC [{args.category}]: N/A "
                f"(no anomaly masks in test set)"
            )
        else:
            pixel_auc = roc_auc_score(pixel_targets_arr, pixel_preds_arr)
            print(f"Pixel ROC-AUC [{args.category}]: {pixel_auc:.4f}")

    # Threshold by Youden J statistic
    fpr, tpr, thresholds = roc_curve(labels, scores)
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    print(f"Threshold (Youden J): {best_thresh:.6f}")

    # Visualisation of the top 3 anomalies
    _plot_anomalies(images, recons, heatmaps, scores, labels, best_thresh, args)


def _plot_anomalies(images, recons, heatmaps, scores, labels, threshold, args):
    # Select the 3 most anomalous images
    anomaly_idx = np.argsort(scores)[::-1][:3]

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle(f"VisA — {args.category}  |  threshold={threshold:.4f}", fontsize=13)

    col_titles = ["Original", "Reconstruction", "Error heatmap"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=10)

    for row, idx in enumerate(anomaly_idx):
        img = images[idx].permute(1, 2, 0).numpy()
        hmap = heatmaps[idx]
        tag = "ANOMALY" if labels[idx] == 1 else "normal"

        axes[row, 0].imshow(img)
        axes[row, 0].set_ylabel(f"score={scores[idx]:.4f}\n[{tag}]", fontsize=8)

        recon = recons[idx].permute(1, 2, 0).numpy()
        axes[row, 1].imshow(recon)

        hmap_norm = hmap / (hmap.max() + 1e-8)
        im = axes[row, 2].imshow(hmap_norm, cmap="hot", vmin=0, vmax=1)
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046)

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    out = Path(args.save_dir) / f"{args.category}_anomalies.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"Visualisation saved: {out}")
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--category",   default="pcb1")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root",  default="data")
    p.add_argument("--save_dir",   default="results")
    p.add_argument("--img_size",   type=int, default=256)
    p.add_argument("--latent_dim", type=int, default=128)
    evaluate(p.parse_args())