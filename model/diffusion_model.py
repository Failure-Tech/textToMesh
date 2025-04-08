# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# import os

# # Dataset Definition
# class TextToVoxelDataset(Dataset):
#     def __init__(self, embeddings_folder, voxels_folder):
#         self.embeddings_files = sorted(os.listdir(embeddings_folder))[:5]
#         self.voxel_files = sorted([
#             os.path.join(root, file)
#             for category in os.listdir(voxels_folder)
#             for root, _, files in os.walk(os.path.join(voxels_folder, category))
#             for file in files if file.endswith(".npy")
#         ])[:5]

#         assert len(self.embeddings_files) == len(self.voxel_files), (
#             f"Mismatch in embeddings ({len(self.embeddings_files)}) and voxels ({len(self.voxel_files)})"
#         )

#         self.embeddings_folder = embeddings_folder
#         self.voxels_folder = voxels_folder

#     def __len__(self):
#         return len(self.embeddings_files)

#     def __getitem__(self, idx):
#         embedding_path = os.path.join(self.embeddings_folder, self.embeddings_files[idx])
#         voxel_path = self.voxel_files[idx]

#         embedding = np.load(embedding_path)
#         voxel = np.load(voxel_path)

#         return torch.tensor(embedding, dtype=torch.float32), torch.tensor(voxel, dtype=torch.float32)

# # Sinusoidal timestep embedding
# def get_timestep_embedding(timesteps, dim):
#     half_dim = dim // 2
#     emb = np.log(10000) / (half_dim - 1)
#     emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
#     emb = timesteps[:, None] * emb[None, :]
#     return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

# # Projector for text embeddings
# class TextConditionProjector(nn.Module):
#     def __init__(self, embedding_dim=1024, latent_dim=256):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(embedding_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, latent_dim),
#         )

#     def forward(self, text_embedding):
#         return self.fc(text_embedding)

# # 3D U-Net Architecture
# class UNet3D(nn.Module):
#     def __init__(self, cond_dim=256):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv3d(2, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv3d(32, 64, 3, stride=2, padding=1),
#             nn.ReLU(),
#         )
#         self.middle = nn.Sequential(
#             nn.Conv3d(64 + cond_dim, 64, 3, padding=1),
#             nn.ReLU(),
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv3d(32, 1, 3, padding=1),
#         )

#     # def forward(self, x, t_emb, cond):
#     #     t_emb = t_emb.view(-1, 1, 1, 1, 1).expand_as(x)
#     #     x = torch.cat([x, t_emb], dim=1)
#     #     x = self.encoder(x)
#     #     cond = cond.view(cond.size(0), cond.size(1), 1, 1, 1).expand(-1, -1, *x.shape[2:])
#     #     x = torch.cat([x, cond], dim=1)
#     #     x = self.middle(x)
#     #     x = self.decoder(x)
#     #     return x

#     def forward(self, x, t_emb, cond):
#         # Problem: t_emb shape mismatch with batch size
#         # Instead of expanding t_emb directly, reshape it correctly
        
#         # Add timestep as a channel rather than trying to expand it
#         batch_size = x.shape[0]
#         t_emb = t_emb.view(batch_size, -1)  # Ensure t_emb matches batch dimension
#         t_emb = t_emb[:, :1].view(batch_size, 1, 1, 1, 1)  # Take just one dimension for the channel
#         x = torch.cat([x, t_emb.expand(-1, -1, *x.shape[2:])], dim=1)  # Concat along channel dimension
        
#         x = self.encoder(x)
#         cond = cond.view(cond.size(0), cond.size(1), 1, 1, 1).expand(-1, -1, *x.shape[2:])
#         x = torch.cat([x, cond], dim=1)
#         x = self.middle(x)
#         x = self.decoder(x)
#         return x

# # Diffusion Trainer
# class DiffusionTextToVoxel:
#     def __init__(self, embedding_dim=1024, latent_dim=256, timesteps=1000):
#         self.timesteps = timesteps
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.betas = torch.linspace(1e-4, 0.02, timesteps).to(self.device)
#         self.alphas = 1. - self.betas
#         self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

#         self.projector = TextConditionProjector(embedding_dim, latent_dim).to(self.device)
#         self.unet = UNet3D(cond_dim=latent_dim).to(self.device)
#         self.optimizer = optim.Adam(list(self.projector.parameters()) + list(self.unet.parameters()), lr=1e-4)

#     def q_sample(self, x_start, t, noise):
#         sqrt_alpha = self.alpha_cumprod[t] ** 0.5
#         sqrt_one_minus_alpha = (1 - self.alpha_cumprod[t]) ** 0.5
#         return sqrt_alpha[:, None, None, None, None] * x_start + sqrt_one_minus_alpha[:, None, None, None, None] * noise

#     def train(self, dataloader, epochs=10):
#         for epoch in range(epochs):
#             for emb, vox in dataloader:
#                 emb, vox = emb.to(self.device), vox.to(self.device)
#                 B = emb.size(0)
#                 t = torch.randint(0, self.timesteps, (B,), device=self.device)
#                 noise = torch.randn_like(vox.unsqueeze(1))
#                 x_noisy = self.q_sample(vox.unsqueeze(1), t, noise)

#                 cond = self.projector(emb)
#                 t_emb = get_timestep_embedding(t, cond.size(1))
#                 pred = self.unet(x_noisy, t_emb, cond)

#                 loss = F.mse_loss(pred, noise)
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()

#             print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

#     def sample(self, cond):
#         self.unet.eval()
#         x = torch.randn((1, 1, 64, 64, 64), device=self.device)
#         cond = self.projector(cond.to(self.device))
#         with torch.no_grad():
#             for t in reversed(range(self.timesteps)):
#                 t_tensor = torch.tensor([t], dtype=torch.long, device=self.device)
#                 t_emb = get_timestep_embedding(t_tensor, cond.size(1))
#                 eps = self.unet(x, t_emb, cond)
#                 alpha = self.alphas[t]
#                 alpha_bar = self.alpha_cumprod[t]
#                 beta = self.betas[t]
#                 x = (1 / alpha**0.5) * (x - (1 - alpha)**0.5 * eps)
#                 if t > 0:
#                     noise = torch.randn_like(x)
#                     x += beta**0.5 * noise
#         return x.squeeze().cpu().numpy()

# # Example dataset and dataloader
# embeddings_folder = "./data/embeddings"
# voxels_folder = "./data/ModelNet40_voxels"
# dataset = TextToVoxelDataset(embeddings_folder, voxels_folder)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# # Trainer
# trainer = DiffusionTextToVoxel()
# trainer.train(dataloader, epochs=10)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# Dataset Definition
class TextToVoxelDataset(Dataset):
    def __init__(self, embeddings_folder, voxels_folder):
        self.embeddings_files = sorted(os.listdir(embeddings_folder))[:5]
        self.voxel_files = sorted([
            os.path.join(root, file)
            for category in os.listdir(voxels_folder)
            for root, _, files in os.walk(os.path.join(voxels_folder, category))
            for file in files if file.endswith(".npy")
        ])[:5]

        assert len(self.embeddings_files) == len(self.voxel_files), (
            f"Mismatch in embeddings ({len(self.embeddings_files)}) and voxels ({len(self.voxel_files)})"
        )

        self.embeddings_folder = embeddings_folder
        self.voxels_folder = voxels_folder

    def __len__(self):
        return len(self.embeddings_files)

    def __getitem__(self, idx):
        embedding_path = os.path.join(self.embeddings_folder, self.embeddings_files[idx])
        voxel_path = self.voxel_files[idx]

        embedding = np.load(embedding_path)
        voxel = np.load(voxel_path)

        return torch.tensor(embedding, dtype=torch.float32), torch.tensor(voxel, dtype=torch.float32)

# Sinusoidal timestep embedding
def get_timestep_embedding(timesteps, dim):
    half_dim = dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb  # Shape: [batch_size, dim]

# Projector for text embeddings
class TextConditionProjector(nn.Module):
    def __init__(self, embedding_dim=1024, latent_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, text_embedding):
        return self.fc(text_embedding)

# 3D U-Net Architecture - FIXED to handle timestep embedding correctly
class UNet3D(nn.Module):
    def __init__(self, cond_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(2, 32, 3, padding=1),  # Input: 1 voxel channel + 1 time channel
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.middle = nn.Sequential(
            nn.Conv3d(64 + cond_dim, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 1, 3, padding=1),
        )

    def forward(self, x, t_emb, cond):
        # Fixed: Add timestep embedding as an additional channel
        batch_size = x.shape[0]
        # Use just the first feature of t_emb as a channel
        t_channel = t_emb[:, :1].view(batch_size, 1, 1, 1, 1).expand(-1, -1, *x.shape[2:])
        x = torch.cat([x, t_channel], dim=1)
        
        x = self.encoder(x)
        cond = cond.view(cond.size(0), cond.size(1), 1, 1, 1).expand(-1, -1, *x.shape[2:])
        x = torch.cat([x, cond], dim=1)
        x = self.middle(x)
        x = self.decoder(x)
        return x

# Diffusion Trainer
class DiffusionTextToVoxel:
    def __init__(self, embedding_dim=1024, latent_dim=256, timesteps=1000):
        self.timesteps = timesteps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.betas = torch.linspace(1e-4, 0.02, timesteps).to(self.device)
        self.alphas = 1. - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

        self.projector = TextConditionProjector(embedding_dim, latent_dim).to(self.device)
        self.unet = UNet3D(cond_dim=latent_dim).to(self.device)
        self.optimizer = optim.Adam(list(self.projector.parameters()) + list(self.unet.parameters()), lr=1e-4)
        
        # Store model configuration
        self.config = {
            "embedding_dim": embedding_dim,
            "latent_dim": latent_dim,
            "timesteps": timesteps
        }

    def q_sample(self, x_start, t, noise):
        sqrt_alpha = self.alpha_cumprod[t] ** 0.5
        sqrt_one_minus_alpha = (1 - self.alpha_cumprod[t]) ** 0.5
        return sqrt_alpha[:, None, None, None, None] * x_start + sqrt_one_minus_alpha[:, None, None, None, None] * noise

    def train(self, dataloader, epochs=10, save_path='text_to_voxel_model.pt'):
        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0
            
            for emb, vox in dataloader:
                emb, vox = emb.to(self.device), vox.to(self.device)
                B = emb.size(0)
                t = torch.randint(0, self.timesteps, (B,), device=self.device)
                noise = torch.randn_like(vox.unsqueeze(1))
                x_noisy = self.q_sample(vox.unsqueeze(1), t, noise)

                cond = self.projector(emb)
                t_emb = get_timestep_embedding(t, cond.size(1))
                pred = self.unet(x_noisy, t_emb, cond)

                loss = F.mse_loss(pred, noise)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1

            avg_loss = epoch_loss / batch_count
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # Save checkpoint after each epoch
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                self.save_model(f"{save_path.split('.')[0]}_epoch{epoch+1}.pt")
        
        # Save final model
        self.save_model(save_path)
        print(f"Model saved to {save_path}")

    def save_model(self, path):
        """Save the model and its configuration"""
        torch.save({
            'projector_state_dict': self.projector.state_dict(),
            'unet_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'betas': self.betas.cpu(),
            'alphas': self.alphas.cpu(),
            'alpha_cumprod': self.alpha_cumprod.cpu()
        }, path)

    def sample(self, cond):
        self.unet.eval()
        x = torch.randn((1, 1, 64, 64, 64), device=self.device)
        cond = self.projector(cond.to(self.device))
        with torch.no_grad():
            for t in reversed(range(self.timesteps)):
                t_tensor = torch.tensor([t], dtype=torch.long, device=self.device)
                t_emb = get_timestep_embedding(t_tensor, cond.size(1))
                eps = self.unet(x, t_emb, cond)
                alpha = self.alphas[t]
                alpha_bar = self.alpha_cumprod[t]
                beta = self.betas[t]
                x = (1 / alpha**0.5) * (x - (1 - alpha)**0.5 * eps)
                if t > 0:
                    noise = torch.randn_like(x)
                    x += beta**0.5 * noise
        return x.squeeze().cpu().numpy()

# Example usage
if __name__ == "__main__":
    # Example dataset and dataloader
    embeddings_folder = "./data/embeddings"
    voxels_folder = "./data/ModelNet40_voxels"
    
    # Create folders if they don't exist 
    os.makedirs(embeddings_folder, exist_ok=True)
    os.makedirs(voxels_folder, exist_ok=True)
    
    dataset = TextToVoxelDataset(embeddings_folder, voxels_folder)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Trainer
    trainer = DiffusionTextToVoxel()
    trainer.train(dataloader, epochs=10, save_path='text_to_voxel_model.pt')