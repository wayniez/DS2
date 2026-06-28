import torch
import torch.nn as nn
 
 
class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
 
        self.enc1 = self._block(3,   32,  stride=2)   # 256 -> 128
        self.enc2 = self._block(32,  64,  stride=2)   # 128 -> 64
        self.enc3 = self._block(64,  128, stride=2)   # 64  -> 32
        self.enc4 = self._block(128, 256, stride=2)   # 32  -> 16
 
        self.project = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, latent_dim),
        )
 
    def _block(self, in_ch, out_ch, stride):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
 
    def forward(self, x):
        x = self.enc1(x)     # [B,  32, 128, 128]
        x = self.enc2(x)     # [B,  64,  64,  64]
        x = self.enc3(x)     # [B, 128,  32,  32]
        x = self.enc4(x)     # [B, 256,  16,  16]
        z = self.project(x)  # [B, latent_dim]
        return z
 
 
class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
 
        self.fc = nn.Linear(latent_dim, 256 * 16 * 16)
    
        self.up1 = self._up(256, 128)   # 16  -> 32
        self.up2 = self._up(128, 64)    # 32  -> 64
        self.up3 = self._up(64,  32)    # 64  -> 128
        self.up4 = self._up(32,  16)    # 128 -> 256
 
        self.final = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
 
    def _up(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
 
    def forward(self, z):
        x = self.fc(z).view(-1, 256, 16, 16)
        x = self.up1(x)        # [B, 128,  32,  32]
        x = self.up2(x)        # [B,  64,  64,  64]
        x = self.up3(x)        # [B,  32, 128, 128]
        x = self.up4(x)        # [B,  16, 256, 256]
        return self.final(x)   # [B, 3, 256, 256]
 
 
class SiameseAutoencoder(nn.Module):
    """
    Takes a pair (x1, x2).
    Train: x1 and x2 are both normal (different augmentations).
    Inference: pass a single image through reconstruct().
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
 
    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        x1_hat = self.decoder(z1)
        x2_hat = self.decoder(z2)
        return x1_hat, x2_hat, z1, z2
 
    def encode(self, x):
        return self.encoder(x)
 
    def reconstruct(self, x):
        z = self.encoder(x)
        return self.decoder(z)