Siamese Autoencoder for Visual Anomaly Detection

Homework 2 - unsupervised anomaly detection on the VisA dataset using a Siamese Autoencoder trained exclusively on normal images.


How it works

The model is trained on pairs of normal images (x1, x2) with different augmentations. The shared encoder is pulled to produce similar latent vectors for both, while the decoder learns to reconstruct each image accurately.

At inference time, a single image is passed through the encoder→decoder pipeline. Anomalous regions produce a high per-pixel reconstruction error, which becomes the anomaly score.

Total loss:

total_loss = recon_loss + α * contrastive_loss


recon_loss - MSE between input and reconstruction (how well the decoder reconstructs the image)
contrastive_loss - MSE between z1 and z2 (how similar the latent representations of two normal images are)
α - weighting coefficient (default: 0.5)


Anomaly score - mean of the top 10% of per-pixel squared errors. The main evaluation metric is ROC-AUC.


Project structure

hw2/
├── model.py      - Siamese Autoencoder (Encoder + Decoder)
├── dataset.py    - VisA dataset loader with pair/single modes
├── losses.py     - ReconLoss, ContrastiveLoss, TotalLoss
├── train.py      - training loop with early stopping and checkpointing
└── evaluate.py   - ROC-AUC evaluation and anomaly visualisation


Dataset

VisA (Visual Anomaly) - 12 industrial categories:
candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum

Expected directory layout:

data/
└── VisA/
    ├── split_csv/
    │   └── 1cls.csv
    ├── pcb1/
    │   ├── Data/
    │   │   ├── Images/Normal/
    │   │   └── Images/Anomaly/
    │   └── Masks/Anomaly/
    └── ...

Download: VisA on Kaggle


Installation

bashpip install torch torchvision scikit-learn matplotlib pandas tqdm pillow


Training

bashpython train.py --category pcb1 --data_root data/VisA --epochs 50

Key arguments:

ArgumentDefaultDescription--categorypcb1VisA category to train on--data_rootdata/VisAPath to dataset root--save_dircheckpointsWhere to save checkpoints--epochs50Number of training epochs--batch_size16Batch size--img_size256Input image resolution--latent_dim128Encoder latent space dimensionality--lr1e-4Learning rate--alpha0.5Weight of contrastive loss--patience10Early stopping patience (epochs)--resumeFalseResume from last checkpoint

The best checkpoint (by ROC-AUC) is saved to checkpoints/<category>_best.pth.


Evaluation

bashpython evaluate.py --category pcb1 --checkpoint checkpoints/pcb1_best.pth

Outputs:


Image-level ROC-AUC - classification of normal vs anomalous images
Pixel-level ROC-AUC - localisation quality (when ground-truth masks are available)
Optimal threshold via Youden J statistic
Visualisation of the top-3 anomalies saved to results/<category>_anomalies.png


Key arguments:

ArgumentDefaultDescription--categorypcb1Category to evaluate--checkpointrequiredPath to .pth checkpoint--data_rootdataPath to dataset root--save_dirresultsWhere to save visualisations


Architecture

Encoder - 4 convolutional blocks (Conv2d → BN → ReLU) with stride 2, followed by AdaptiveAvgPool and a Linear projection to latent_dim.

Input [B, 3, 256, 256]
  → Conv block  [B, 32,  128, 128]
  → Conv block  [B, 64,   64,  64]
  → Conv block  [B, 128,  32,  32]
  → Conv block  [B, 256,  16,  16]
  → AvgPool + Linear → [B, latent_dim]

Decoder - Linear projection back to spatial features, followed by 4 transposed convolution blocks and a final Conv + Sigmoid.

[B, latent_dim]
  → Linear → [B, 256, 16, 16]
  → ConvTranspose  [B, 128, 32,  32]
  → ConvTranspose  [B,  64, 64,  64]
  → ConvTranspose  [B,  32, 128, 128]
  → ConvTranspose  [B,  16, 256, 256]
  → Conv + Sigmoid → [B, 3, 256, 256]
