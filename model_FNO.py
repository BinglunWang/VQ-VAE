import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO
from vector_quantize_pytorch import VectorQuantize


class FNOEncoder(nn.Module):
    def __init__(self, embedding_dim=64, modes=8):
        super().__init__()
        # Project from 1 input channel up to embedding_dim
        self.input_proj = nn.Conv2d(1, embedding_dim, kernel_size=1)

        # FNO processes in frequency domain at full resolution — no spatial info lost
        self.fno = FNO(
            n_modes=(modes, modes),
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            hidden_channels=8,
            n_layers=2,
        )

        # Single downsample at the end: 28x28 -> 14x14
        # Keeps much more spatial info than two strided convs (28->14->7)
        self.downsample = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.input_proj(x)   # (B, embedding_dim, 28, 28)
        x = self.fno(x)          # (B, embedding_dim, 28, 28)
        x = self.downsample(x)   # (B, embedding_dim, 14, 14)
        return x


class FNODecoder(nn.Module):
    def __init__(self, embedding_dim=64, modes=8, output_size=56):
        """
        Args:
            output_size: target output resolution.
                         28 = same as input, 56/112 for super-resolution.
        """
        super().__init__()
        self.output_size = output_size

        # FNO learns the reconstruction mapping at full target resolution
        self.fno = FNO(
            n_modes=(modes, modes),
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            hidden_channels=8,
            n_layers=2,
        )

        # Project down to grayscale output
        self.output_proj = nn.Conv2d(embedding_dim, 1, kernel_size=1)

    def forward(self, x):
        # Upsample first so FNO learns reconstruction at full target resolution
        # rather than upsampling a blurry low-res output at the end
        x = F.interpolate(x, size=(self.output_size, self.output_size),
                          mode='bilinear', align_corners=False)  # (B, C, output_size, output_size)
        x = self.fno(x)          # (B, embedding_dim, output_size, output_size)
        x = self.output_proj(x)  # (B, 1, output_size, output_size)
        x = torch.sigmoid(x)     # bound to [0, 1]
        return x


class VQVAE(nn.Module):
    def __init__(self, embedding_dim=64, num_embeddings=256, modes=8, output_size=56):
        """
        Args:
            embedding_dim: channel dimension for encoder/decoder and codebook
            num_embeddings: codebook size
            modes: number of Fourier modes in FNO layers
                   (higher = more detail, but more memory; 8 is good for 28x28 input)
            output_size: decoder output resolution (28=same, 56/112=super-res)
        """
        super().__init__()
        self.output_size = output_size

        self.encoder = FNOEncoder(embedding_dim=embedding_dim, modes=modes)
        self.vq = VectorQuantize(
            dim=embedding_dim,
            codebook_size=num_embeddings,
            decay=0.99,
            commitment_weight=1.0,  # no pre-scaling; compute_loss owns it
        )
        self.decoder = FNODecoder(embedding_dim=embedding_dim, modes=modes, output_size=output_size)

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
            x:             Original input images (28x28)
            x_hat:         Reconstructed images (output_size x output_size)
            commit_loss:   Raw commitment loss from VQ layer (unscaled)
            recon_weight:  Weight for reconstruction loss
            commit_weight: Weight for commitment loss (sole owner of scaling)

        Returns:
            Dictionary with total_loss, recon_loss, commit_loss
        """
        # Upsample ground truth to match decoder output resolution before MSE
        x_up = F.interpolate(x, size=(self.output_size, self.output_size),
                             mode='bilinear', align_corners=False)
        recon_loss = F.mse_loss(x_hat, x_up)
        total_loss = recon_weight * recon_loss + commit_weight * commit_loss

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'commit_loss': commit_loss,
        }