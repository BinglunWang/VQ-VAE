from model import VQVAE as CNN_VQVAE
from model_FNO import VQVAE as FNO_VQVAE # Assuming you saved them in different files

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

cnn_model = CNN_VQVAE(embedding_dim=8, num_embeddings=8)
fno_model = FNO_VQVAE(embedding_dim=8, num_embeddings=8, modes=4)

print(f"CNN Parameters: {count_parameters(cnn_model):,}")
print(f"FNO Parameters: {count_parameters(fno_model):,}")