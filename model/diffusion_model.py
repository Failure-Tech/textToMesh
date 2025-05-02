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
import random

# Dataset Definition
class TextToVoxelDataset(Dataset):
    def __init__(self, embeddings_folder, voxels_folder):
        self.embeddings = [
            os.path.join(embeddings_folder, f)
            for f in os.listdir(embeddings_folder)
            if f.endswith(".npy")
        ]
        assert len(self.embeddings) > 0, "No embedding files found!"

        self.voxels = []
        for root, _, files in os.walk(voxels_folder):
            for file in files:
                if file.endswith(".npy"):
                    self.voxels.append(os.path.join(root, file))

        assert len(self.voxels) > 0, "No voxel files found!"

        print(f"📦 Loaded {len(self.voxels)} voxel files")
        print(f"🔤 Loaded {len(self.embeddings)} embedding files")

    def __len__(self):
        return len(self.voxels)

    def __getitem__(self, idx):
        voxel_path = self.voxels[idx]
        emb_path = random.choice(self.embeddings)

        voxel = np.load(voxel_path).astype(np.float32)
        embedding = np.load(emb_path).astype(np.float32)

        # Return as torch tensors
        return (
            torch.from_numpy(embedding),
            torch.from_numpy(voxel)
        )
    
# Sinusoidal timestep embedding
def get_timestep_embedding(timesteps, dim):
    half_dim = dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:  # If dim is odd, add a zero column
        emb = torch.cat([emb, torch.zeros((timesteps.shape[0], 1), device=timesteps.device)], dim=1)
    return emb

# Improved Text Condition Projector
class TextConditionProjector(nn.Module):
    def __init__(self, embedding_dim=1024, latent_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, text_embedding):
        return self.fc(text_embedding)

# Resblock for 3D convolutions
class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, temb_dim=256, downsample=False):
        super().__init__()
        self.downsample = downsample
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        
        # Skip connection handling
        if in_channels != out_channels or downsample:
            self.skip = nn.Conv3d(in_channels, out_channels, 1, stride=stride)
        else:
            self.skip = nn.Identity()
            
        # Conditioning projection
        self.temb_proj = nn.Linear(temb_dim, out_channels)
            
    def forward(self, x, temb=None):
        h = self.act(self.norm1(self.conv1(x)))
        
        # Add time embedding
        if temb is not None:
            h = h + self.temb_proj(temb)[:, :, None, None, None]
            
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)

# 3D Attention Block
class Attention3D(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj = nn.Conv3d(channels, channels, 1)
        
    def forward(self, x):
        b, c, d, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).reshape(b, 3, self.num_heads, c // self.num_heads, d * h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Attention
        attn = torch.einsum('bhnc,bhmc->bhnm', q, k) * (c // self.num_heads) ** -0.5
        attn = F.softmax(attn, dim=-1)
        
        # Attend to values
        out = torch.einsum('bhnm,bhmc->bhnc', attn, v)
        out = out.reshape(b, c, d, h, w)
        return x + self.proj(out)

# Improved 3D U-Net with residual blocks, attention, and better conditioning
class UNet3D(nn.Module):
    def __init__(self, time_dim=256, cond_dim=256, channels=[32, 64, 128, 256]):
        super().__init__()
        # Make sure time_dim matches the conditioning dim
        self.time_dim = time_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Conditioning projection - make sure output matches time_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Initial projection
        self.init_conv = nn.Conv3d(1, channels[0], kernel_size=3, padding=1)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        input_channels = channels[0]
        for i in range(len(channels)):
            # Skip the first channel as we already have init_conv
            out_channels = channels[i]
            down_block = nn.ModuleList([
                ResBlock3D(input_channels, out_channels, time_dim),
                ResBlock3D(out_channels, out_channels, time_dim),
                Attention3D(out_channels) if i > 0 else nn.Identity(),  # Attention for deeper layers
                nn.Conv3d(out_channels, out_channels, 3, stride=2, padding=1) if i < len(channels) - 1 else nn.Identity()
            ])
            self.down_blocks.append(down_block)
            input_channels = out_channels
        
        # Middle
        self.middle_block = nn.ModuleList([
            ResBlock3D(channels[-1], channels[-1], time_dim),
            Attention3D(channels[-1]),
            ResBlock3D(channels[-1], channels[-1], time_dim)
        ])
        
        # Upsampling path
        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        for i in range(len(reversed_channels) - 1):
            up_block = nn.ModuleList([
                nn.ConvTranspose3d(reversed_channels[i], reversed_channels[i], 4, stride=2, padding=1),
                ResBlock3D(reversed_channels[i] + reversed_channels[i + 1], reversed_channels[i + 1], time_dim),
                ResBlock3D(reversed_channels[i + 1], reversed_channels[i + 1], time_dim),
                Attention3D(reversed_channels[i + 1]) if i < len(reversed_channels) - 2 else nn.Identity()
            ])
            self.up_blocks.append(up_block)
        
        # Final layers
        self.final_block = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv3d(channels[0], 1, kernel_size=3, padding=1)
        )
        
    def forward(self, x, t_emb, cond):
        # Process timestep and condition embeddings separately first
        t = self.time_mlp(t_emb)
        c = self.cond_proj(cond)
        
        # Combine embeddings - both should now be the same dimension
        temb = t + c
        
        # Initial conv
        h = self.init_conv(x)
        
        # Store skip connections
        skips = []
        
        # Downsampling
        for i, (res1, res2, attn, downsample) in enumerate(self.down_blocks):
            h = res1(h, temb)
            h = res2(h, temb)
            h = attn(h)
            skips.append(h)
            h = downsample(h)
        
        # Middle
        h = self.middle_block[0](h, temb)
        h = self.middle_block[1](h)
        h = self.middle_block[2](h, temb)
        
        # Upsampling
        for i, (upsample, res1, res2, attn) in enumerate(self.up_blocks):
            # Get skip connection
            skip = skips.pop()
            
            # Upsample
            h = upsample(h)
            
            # Concatenate with skip connection
            h = torch.cat([h, skip], dim=1)
            
            # Process
            h = res1(h, temb)
            h = res2(h, temb)
            h = attn(h)
        
        # Final layers
        return self.final_block(h)

# Improved Diffusion Trainer with better sampling
class DiffusionTextToVoxel:
    def __init__(self, embedding_dim=1024, latent_dim=256, timesteps=1000):
        self.timesteps = timesteps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.betas = torch.linspace(1e-4, 0.02, timesteps).to(self.device)
        self.alphas = 1. - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)
        
        # Create models with matching dimensions
        self.projector = TextConditionProjector(embedding_dim, latent_dim).to(self.device)
        self.unet = UNet3D(time_dim=latent_dim, cond_dim=latent_dim).to(self.device)
        
        # Using AdamW with a smaller learning rate for better training
        self.optimizer = optim.AdamW(
            list(self.projector.parameters()) + list(self.unet.parameters()), 
            lr=2e-5, 
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        
        self.config = {
            "embedding_dim": embedding_dim,
            "latent_dim": latent_dim,
            "timesteps": timesteps
        }

    def q_sample(self, x_start, t):
        noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alpha_cumprod[t][:, None, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[t][:, None, None, None, None]
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise, noise

    def train(self, dataloader, epochs=10, save_every=5, save_path='text_to_voxel_model.pt'):
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.unet.train()
            self.projector.train()
            epoch_loss = 0
            
            for batch_idx, (emb, vox) in enumerate(dataloader):
                emb, vox = emb.to(self.device), vox.to(self.device)
                B = emb.size(0)
                
                # Sample timesteps
                t = torch.randint(0, self.timesteps, (B,), device=self.device)
                
                # Add noise to voxels
                x_noisy, noise = self.q_sample(vox.unsqueeze(1), t)
                
                # Get timestep embeddings
                t_emb = get_timestep_embedding(t, self.config["latent_dim"])
                
                # Get conditioning
                cond = self.projector(emb)
                
                # Predict noise
                noise_pred = self.unet(x_noisy, t_emb, cond)
                
                # Calculate loss
                loss = F.mse_loss(noise_pred, noise)
                
                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    list(self.projector.parameters()) + list(self.unet.parameters()), 
                    max_norm=1.0
                )
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # Print batch progress every 20 batches
                if batch_idx % 20 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")
            
            # Step LR scheduler
            self.scheduler.step()
            
            # Calculate average epoch loss
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}")
            
            # Save periodically and when loss improves
            if (epoch + 1) % save_every == 0 or avg_loss < best_loss:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.save_model(f"{save_path.split('.')[0]}_best.pt")
                
                self.save_model(f"{save_path.split('.')[0]}_epoch{epoch+1}.pt")
        
        # Final save
        self.save_model(save_path)

    def save_model(self, path):
        torch.save({
            'projector_state_dict': self.projector.state_dict(),
            'unet_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'betas': self.betas.cpu(),
            'alphas': self.alphas.cpu(),
            'alpha_cumprod': self.alpha_cumprod.cpu()
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.projector.load_state_dict(checkpoint['projector_state_dict'])
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.config = checkpoint['config']
        self.betas = checkpoint['betas'].to(self.device)
        self.alphas = checkpoint['alphas'].to(self.device)
        self.alpha_cumprod = checkpoint['alpha_cumprod'].to(self.device)
        print(f"Model loaded from {path}")

    # Improved sampling with DDIM for faster, better quality sampling
    def sample(self, cond, steps=100, eta=0.0):
        """
        DDIM sampling for faster generation with better quality
        Args:
            cond: text conditioning
            steps: number of sampling steps (fewer than training timesteps)
            eta: 0 = deterministic, 1 = stochastic (like DDPM)
        """
        self.unet.eval()
        self.projector.eval()
        
        # Choose which timesteps to sample
        times = torch.linspace(0, self.timesteps - 1, steps + 1).round().long().flip(0).to(self.device)
        
        # Initialize with random noise
        x = torch.randn((1, 1, 64, 64, 64), device=self.device)
        
        # Process text embedding
        with torch.no_grad():
            cond = self.projector(cond.to(self.device))
        
        # Sampling loop
        with torch.no_grad():
            for i in range(len(times) - 1):
                # Current and next timestep
                t_curr = times[i]
                t_next = times[i + 1]
                
                # Get embeddings
                t_emb = get_timestep_embedding(t_curr.unsqueeze(0), self.config["latent_dim"])
                
                # Predict noise
                eps = self.unet(x, t_emb, cond)
                
                # Get alpha values
                alpha_curr = self.alpha_cumprod[t_curr]
                alpha_next = self.alpha_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0).to(self.device)
                
                # Current x without noise
                x0_pred = (x - torch.sqrt(1 - alpha_curr) * eps) / torch.sqrt(alpha_curr)
                
                # Direction pointing to x_t
                dir_xt = torch.sqrt(1 - alpha_next - eta**2 * (1 - alpha_next) / (1 - alpha_curr)) * eps
                
                # Random noise for stochasticity
                if eta > 0:
                    noise = eta * torch.randn_like(x)
                    x = torch.sqrt(alpha_next) * x0_pred + dir_xt + torch.sqrt(1 - alpha_next) * noise
                else:
                    x = torch.sqrt(alpha_next) * x0_pred + dir_xt
                
        # Final result
        return x.squeeze().cpu().numpy()

# Example usage
if __name__ == "__main__":
    embeddings_folder = "./data/embeddings"
    voxels_folder = "./data/ModelNet40_voxels"

    os.makedirs(embeddings_folder, exist_ok=True)
    os.makedirs(voxels_folder, exist_ok=True)

    try:
        dataset = TextToVoxelDataset(embeddings_folder, voxels_folder)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        trainer = DiffusionTextToVoxel()
        trainer.train(dataloader, epochs=50, save_every=5, save_path='text_to_voxel_model.pt')
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()