import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, embedding_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class VQVAE(nn.Module):
    def __init__(self, embedding_dim=64, num_embeddings=256):
        super().__init__()
        self.encoder = Encoder(embedding_dim)
        self.vq = VectorQuantize(
            dim=embedding_dim,
            codebook_size=num_embeddings,
            decay=0.99,
            commitment_weight=1.0,  # Fixed: no pre-scaling; compute_loss owns the weight
        )
        self.decoder = Decoder(embedding_dim)

    def encode(self, x):
        """Encode input image to quantized latent representation"""
        B = x.shape[0]
        z_e = self.encoder(x)                           # (B, C, H, W)
        z_e_t = z_e.permute(0, 2, 3, 1)                # (B, H, W, C)
        H, W = z_e_t.shape[1], z_e_t.shape[2]

        z_e_flat = z_e_t.reshape(B * H * W, -1)        # (B*H*W, C)
        z_q_flat, indices, commit_loss = self.vq(z_e_flat)

        z_q = z_q_flat.reshape(B, H, W, -1)            # (B, H, W, C)
        z_q_t = z_q.permute(0, 3, 1, 2)                # (B, C, H, W)
        return z_q_t, indices, commit_loss

    def decode(self, z_q):
        """Decode latent representation to image"""
        return self.decoder(z_q)  # expects (B, C, H, W)

    def forward(self, x):
        """Forward pass through the entire VQ-VAE"""
        B = x.shape[0]
        z_e = self.encoder(x)                           # (B, C, H, W)
        z_e_t = z_e.permute(0, 2, 3, 1)                # (B, H, W, C)
        H, W = z_e_t.shape[1], z_e_t.shape[2]

        z_e_flat = z_e_t.reshape(B * H * W, -1)        # (B*H*W, C)
        z_q_flat, indices, commit_loss = self.vq(z_e_flat)

        z_q = z_q_flat.reshape(B, H, W, -1)            # (B, H, W, C)
        z_q_t = z_q.permute(0, 3, 1, 2)                # (B, C, H, W)

        x_hat = self.decoder(z_q_t)
        return x_hat, indices, commit_loss

    def compute_loss(self, x, x_hat, commit_loss, recon_weight=1.0, commit_weight=0.25):
        recon_loss = F.mse_loss(x_hat, x)
        total_loss = recon_weight * recon_loss + commit_weight * commit_loss  # sole owner of commit scaling
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'commit_loss': commit_loss
        }