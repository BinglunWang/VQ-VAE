import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            # (B, 1, 28, 28) -> (B, 32, 14, 14)
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # (B, 32, 14, 14) -> (B, embedding_dim, 7, 7)
            nn.Conv2d(32, embedding_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            # (B, embedding_dim, 7, 7) -> (B, 32, 14, 14)
            nn.ConvTranspose2d(embedding_dim, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # (B, 32, 14, 14) -> (B, 1, 28, 28)
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
            commitment_weight=1.0,  # no pre-scaling; compute_loss owns it
        )
        self.decoder = Decoder(embedding_dim)

    def encode(self, x):
        """Encode input image to quantized latent representation"""
        B = x.shape[0]
        z_e = self.encoder(x)                        # (B, C, H, W)
        z_e_t = z_e.permute(0, 2, 3, 1)             # (B, H, W, C)
        H, W = z_e_t.shape[1], z_e_t.shape[2]

        z_e_flat = z_e_t.reshape(B * H * W, -1)     # (B*H*W, C)
        z_q_flat, indices, commit_loss = self.vq(z_e_flat)

        z_q = z_q_flat.reshape(B, H, W, -1)         # (B, H, W, C)
        z_q_t = z_q.permute(0, 3, 1, 2)             # (B, C, H, W)
        return z_q_t, indices, commit_loss

    def decode(self, z_q):
        """Decode latent representation to image — expects (B, C, H, W)"""
        return self.decoder(z_q)

    def forward(self, x):
        """Forward pass through the entire VQ-VAE"""
        B = x.shape[0]

        # 1. Encode
        z_e = self.encoder(x)                        # (B, C, H, W)

        # 2. Reshape for VQ layer: (B, C, H, W) -> (B*H*W, C)
        z_e_t = z_e.permute(0, 2, 3, 1)             # (B, H, W, C)
        H, W = z_e_t.shape[1], z_e_t.shape[2]
        z_e_flat = z_e_t.reshape(B * H * W, -1)     # (B*H*W, C)

        # 3. Quantize
        z_q_flat, indices, commit_loss = self.vq(z_e_flat)

        # 4. Reshape back for decoder: (B*H*W, C) -> (B, C, H, W)
        z_q = z_q_flat.reshape(B, H, W, -1)         # (B, H, W, C)
        z_q_t = z_q.permute(0, 3, 1, 2)             # (B, C, H, W)

        # 5. Decode
        x_hat = self.decoder(z_q_t)

        return x_hat, indices, commit_loss

    def compute_loss(self, x, x_hat, commit_loss, recon_weight=1.0, commit_weight=0.25):
        """
        Compute the total VQ-VAE loss.

        Args:
            x:             Original input images
            x_hat:         Reconstructed images
            commit_loss:   Raw commitment loss from VQ layer (unscaled)
            recon_weight:  Weight for reconstruction loss
            commit_weight: Weight for commitment loss (sole owner of scaling)

        Returns:
            Dictionary with total_loss, recon_loss, commit_loss
        """
        recon_loss = F.mse_loss(x_hat, x)
        total_loss = recon_weight * recon_loss + commit_weight * commit_loss

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'commit_loss': commit_loss,
        }