"""
Training Siamese Autoencoder on VisA.

total_loss = recon_loss + alpha * contrast_loss   <---- the loss function that the model optimizes
recon_loss = ReconLoss(x1, x1_hat)   <---- indicates how well the decoder reconstructs the input image from the latent vector.
total_contrast = MSE(z1, z2)   <----  contrastive loss between latent vectors, Indicates how similar the latent representations 
                                      of two normal images are. The smaller the value, the closer z1 and z2 are in latent space.

main metric - auc <---- model is saved every auc increase of previous best score 

Example:

python train.py --category pcb1 --data_root data/VisA --epochs 50
"""

import argparse
from pathlib import Path
import numpy as np
 
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
 
from dataset import build_dataloaders
from model import SiameseAutoencoder
from losses import TotalLoss
from sklearn.metrics import roc_auc_score
 
 
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
 
    total_loss = 0.0
    total_recon = 0.0
    total_contrast = 0.0
 
    pbar = tqdm(loader, leave=False)
 
    for batch in pbar:
        x1 = batch["x1"].to(device)
        x2 = batch["x2"].to(device)
 
        optimizer.zero_grad()
        x1_hat, x2_hat, z1, z2 = model(x1, x2)
        loss, recon_loss, contrast_loss = criterion(x1, x2, x1_hat, x2_hat, z1, z2)
 
        loss.backward()
        optimizer.step()
 
        total_loss += loss.item()
        total_recon += recon_loss
        total_contrast += contrast_loss
 
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            recon=f"{recon_loss:.4f}",
            contrast=f"{contrast_loss:.4f}",
        )
 
    n = len(loader)
    return total_loss / n, total_recon / n, total_contrast / n
 
 
def save_checkpoint(model, optimizer, scheduler, epoch, best_auc, latent_dim, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":      epoch,
        "best_auc":   best_auc,
        "latent_dim": latent_dim, 
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict(),
    }, path)
 
 
def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["best_auc"]
 
 
@torch.no_grad()
def evaluate_auc(model, loader, device):
    model.eval()
    scores = []
    labels = []
 
    for batch in loader:
        x = batch["image"].to(device)
        x_hat = model.reconstruct(x)
        err = (x - x_hat).pow(2).mean(dim=1)  # [B, H, W]
        # top-k average: take the 10% of pixels with the largest error
        # better at detecting local defects than the global average
        k = max(1, int(err.shape[-1] * err.shape[-2] * 0.1))
        score = err.flatten(1).topk(k, dim=1).values.mean(dim=1)
        scores.extend(score.cpu().numpy())
        labels.extend(batch["label"].cpu().numpy())
 
    scores = np.asarray(scores)
    labels = np.asarray(labels)
 
    if len(np.unique(labels)) < 2:
        return 0.5
 
    return roc_auc_score(labels, scores)
 
 
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.stop = False
 
    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
 
        improvement = score - self.best_score
 
        if improvement > self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True
 
        return self.stop
 
 
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
 
    train_loader, test_loader = build_dataloaders(
        root=args.data_root,
        category=args.category,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
 
    model = SiameseAutoencoder(latent_dim=args.latent_dim).to(device)
    criterion = TotalLoss(alpha=args.alpha, ssim_weight=args.ssim_weight)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
 
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",        # Monitor the AUC (higher is better)
        factor=0.5,        # lr *= 0.5 when it plateaus
        patience=5,        # Wait 5 epochs without improvement
        min_lr=1e-6,       # Lower bound for lr
    )
 
    early_stopping = EarlyStopping(patience=args.patience)
 
    best_auc    = 0.0
    start_epoch = 0
    checkpoint_dir = Path(args.save_dir)
    best_path = checkpoint_dir / f"{args.category}_best.pth"
 
    if args.resume:
        if best_path.exists():
            start_epoch, best_auc = load_checkpoint(
                best_path, model, optimizer, scheduler, device
            )
            start_epoch += 1 
            early_stopping.best_score = best_auc
            print(f"Resumed from epoch {start_epoch}, best_auc={best_auc:.4f}")
        else:
            print(f"[Warning] --resume specified, but the checkpoint was not found: {best_path}")
            print("Starting training from scratch.")
 
    for epoch in range(start_epoch, args.epochs):
 
        loss, recon, contrast = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
 
        auc = evaluate_auc(model, test_loader, device)
 
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"loss={loss:.6f} "
            f"recon={recon:.6f} "
            f"contrast={contrast:.6f} "
            f"auc={auc:.4f} "
            f"lr={current_lr:.2e}"
        )
 
        scheduler.step(auc)
 
        if auc > best_auc:
            best_auc = auc
            save_checkpoint(model, optimizer, scheduler, epoch, best_auc, args.latent_dim, best_path)
            print(f"  -> best model saved (auc={best_auc:.4f})")
 
        if early_stopping.step(auc):
            print("\nEarly stopping triggered.")
            break
 
    print("\nTraining finished.")
    print(f"Best AUC: {best_auc:.4f}")
    print(f"Checkpoint: {best_path}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category",     default="pcb1")
    parser.add_argument("--data_root",    default="data/VisA")
    parser.add_argument("--save_dir",     default="checkpoints")
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--img_size",     type=int,   default=256)
    parser.add_argument("--latent_dim",   type=int,   default=128)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--alpha",        type=float, default=0.5)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--patience",     type=int,   default=10)
    parser.add_argument("--ssim_weight",  type=float, default=0.5,
                        help="SSIM weight in the recon loss (0 = pure MSE, 0.5 = 50/50)")
    parser.add_argument("--resume",       action="store_true",
                        help="continue the training from the last checkpoint")
    args = parser.parse_args()
    main(args)