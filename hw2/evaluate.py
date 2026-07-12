import argparse
from pathlib import Path
 
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
 
from dataset import build_dataloaders
from model import SiameseAutoencoder


def _smooth_heatmap(hmap, sigma=3):
    """Gaussian blur to suppress edge artifacts"""
    k = 2 * sigma * 3 + 1
    x = torch.arange(k, dtype=torch.float32) - k // 2
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    gauss /= gauss.sum()
    kernel = gauss.outer(gauss).unsqueeze(0).unsqueeze(0).to(hmap.device)

    hmap = hmap.unsqueeze(1)  # [B, 1, H, W]
    hmap = F.conv2d(hmap, kernel, padding=k // 2)
    return hmap.squeeze(1)


def multiscale_heatmap(x, x_hat, features_x, features_xhat):
    """
    Multi-scale heatmap: combines the reconstruction error
    in the output image and in the encoder's intermediate layers.

    The idea: a defect may be visible at different scales.
    Small defects are more visible on high-resolution feature maps (f1, f2),
    while large defects are more visible at the output (x vs x_hat).
    """
    H, W = x.shape[-2], x.shape[-1]

    # Error in the output image [B, H, W]
    pixel_err = (x - x_hat).pow(2).mean(dim=1)

    # Errors in the encoder's intermediate feature maps
    feat_errors = []
    for f_x, f_xhat in zip(features_x, features_xhat):
        err = (f_x - f_xhat).pow(2).mean(dim=1, keepdim=True)  # [B, 1, h, w]
        # We upscaling to the size of the output image
        err_up = F.interpolate(
            err, size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(1)  # [B, H, W]
        feat_errors.append(err_up)

    # [pixel, z, f1, f2, f3, f4]
    weights = [0.5, 0.4, 0.05, 0.05, 0.2, 0.3]
    heatmap = weights[0] * pixel_err
    for w, fe in zip(weights[1:], feat_errors):
        heatmap = heatmap + w * fe

    # Gaussian blur — suppresses edge artifacts (subpixel shifts)
    heatmap = _smooth_heatmap(heatmap, sigma=3)

    # Two scoring variants — which one wins depends on whether the raw
    # reconstruction-error magnitude tracks anomalies (favors "raw") or is
    # swamped by per-image nuisance factors like lighting/color drift
    # (favors "norm", which cancels out each image's own dynamic range).
    B = heatmap.shape[0]
    k = max(1, int(H * W * 0.1))

    score_raw = heatmap.flatten(1).topk(k, dim=1).values.mean(dim=1)

    hmap_min = heatmap.flatten(1).min(1).values.view(B, 1, 1)
    hmap_max = heatmap.flatten(1).max(1).values.view(B, 1, 1)
    heatmap_norm = (heatmap - hmap_min) / (hmap_max - hmap_min + 1e-8)
    score_norm = heatmap_norm.flatten(1).topk(k, dim=1).values.mean(dim=1)

    # heatmap_norm doubles as the visualization heatmap (0..1 for the colormap)
    return score_raw, score_norm, heatmap_norm


def denormalize(t):
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

    ckpt = torch.load(args.checkpoint, map_location=device)
    latent_dim = ckpt.get("latent_dim", args.latent_dim)
    model = SiameseAutoencoder(latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    scores_raw = []
    scores_norm = []
    labels = []
    heatmaps = []
    images = []
    recons = []
    pixel_preds = []
    pixel_targets = []
    image_paths = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["image"].to(device)

            z_x,    features_x    = model.encode(x)
            x_hat,  features_xhat = model.reconstruct_with_features(x)
            z_xhat, _             = model.encode(x_hat)

            features_x    = [z_x]    + features_x
            features_xhat = [z_xhat] + features_xhat

            score_raw, score_norm, hmap = multiscale_heatmap(x, x_hat, features_x, features_xhat)

            scores_raw.append(score_raw.item())
            scores_norm.append(score_norm.item())
            labels.append(batch["label"].item())
            heatmaps.append(hmap.squeeze().cpu().numpy())
            images.append(denormalize(x.squeeze()))
            recons.append(denormalize(x_hat.squeeze()))
            image_paths.append(batch["image_path"][0])

            mask = batch["mask"]
            if batch["has_mask"][0]:
                pixel_preds.extend(hmap.flatten().cpu().numpy())
                pixel_targets.extend(
                    (mask.flatten().cpu().numpy() > 0.5).astype(int)
                )

    scores_raw = np.array(scores_raw)
    scores_norm = np.array(scores_norm)
    labels = np.array(labels)

    auc_raw = roc_auc_score(labels, scores_raw)
    auc_norm = roc_auc_score(labels, scores_norm)
    print(f"\nROC-AUC [{args.category}] raw (unnormalized magnitude):  {auc_raw:.4f}")
    print(f"ROC-AUC [{args.category}] norm (per-image normalized):    {auc_norm:.4f}")

    # Use whichever scoring variant wins for downstream diagnostics/plot
    if auc_raw >= auc_norm:
        print(f"-> using RAW scoring (better AUC)")
        scores = scores_raw
    else:
        print(f"-> using NORM scoring (better AUC)")
        scores = scores_norm

    # --- Duplicate diagnostics ---
    # 1) Exact duplicate image paths in the test split (dataset/CSV issue)
    seen = {}
    for p in image_paths:
        seen[p] = seen.get(p, 0) + 1
    dup_paths = {p: c for p, c in seen.items() if c > 1}
    if dup_paths:
        print(f"\n[WARNING] Found {len(dup_paths)} duplicate image path(s) in test split:")
        for p, c in dup_paths.items():
            print(f"  {p}  (appears {c} times)")
    else:
        print("\nNo duplicate image paths found in test split.")

    # 2) Samples with (near-)identical scores but different image paths
    #    — flags coincidental ties worth a manual look
    order = np.argsort(scores)
    tie_groups = []
    i = 0
    while i < len(order) - 1:
        j = i
        while j + 1 < len(order) and abs(scores[order[j + 1]] - scores[order[i]]) < 1e-4:
            j += 1
        if j > i:
            tie_groups.append(order[i:j + 1])
        i = j + 1
    if tie_groups:
        print(f"\n[INFO] Found {len(tie_groups)} group(s) of near-identical scores (tol=1e-4):")
        for g in tie_groups:
            for idx in g:
                print(f"  score={scores[idx]:.4f}  label={labels[idx]}  path={image_paths[idx]}")
            print("  ---")

    if len(pixel_targets) > 0:
        pixel_targets_arr = np.array(pixel_targets)
        pixel_preds_arr   = np.array(pixel_preds)
        if len(np.unique(pixel_targets_arr)) < 2:
            print(f"Pixel ROC-AUC [{args.category}]: N/A (no anomaly masks in test set)")
        else:
            pixel_auc = roc_auc_score(pixel_targets_arr, pixel_preds_arr)
            print(f"Pixel ROC-AUC [{args.category}]: {pixel_auc:.4f}")

    fpr, tpr, thresholds = roc_curve(labels, scores)
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    print(f"Threshold (Youden J): {best_thresh:.6f}")

    _plot_anomalies(images, recons, heatmaps, scores, labels, best_thresh, args, image_paths)


def _plot_anomalies(images, recons, heatmaps, scores, labels, threshold, args, image_paths=None):
    anomaly_idx = np.argsort(scores)[::-1][:3]

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle(f"VisA — {args.category}  |  threshold={threshold:.4f}", fontsize=13)

    col_titles = ["Original", "Reconstruction", "Multiscale heatmap"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=10)

    for row, idx in enumerate(anomaly_idx):
        img  = images[idx].permute(1, 2, 0).numpy()
        hmap = heatmaps[idx]
        tag  = "ANOMALY" if labels[idx] == 1 else "normal"

        path_str = Path(image_paths[idx]).name if image_paths is not None else ""
        axes[row, 0].imshow(img)
        axes[row, 0].text(
            -0.15, 0.5, f"score={scores[idx]:.4f}\n[{tag}]\n{path_str}",
            transform=axes[row, 0].transAxes,
            ha="right", va="center", fontsize=7
        )

        axes[row, 1].imshow(recons[idx].permute(1, 2, 0).numpy())

        im = axes[row, 2].imshow(hmap, cmap="hot", vmin=0, vmax=1)
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