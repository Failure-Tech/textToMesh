import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataSet, DataLoader
from PIL import Image

class PositionalEncoder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, num_freqs):
        super().__init__()
        self.freq_bands = 2**torch.linspace(0, max_freq_log2, num_freqs)
        self.output_dim = input_dim * (2*num_freqs)

    def forward(self, x):
        encoded = [x]
        for freq in self.freq_bands:
            encoded.append(torch.sin(freq*x))
            encoded.append(torch.cos(freq*x))

        return torch.cat(encoded, dim=1)

class NeRFModel(nn.Module):
    def __init__(self, pos_encoder, dir_encoder):
        super().__init__()
        self.pos_encoder = pos_encoder
        self.dir_encoder = dir_encoder

        self.mlp = nn.Sequential(
            nn.Linear(pos_encoder.output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),            
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.sigma_layer = nn.Linear(256, 1)
        self.feature_layer = nn.Linear(256, 256)

        self.color_layer = nn.Sequential(
            nn.Linear(256 + dir_encoder.output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

    def forward(self, pos, dirs):
        pos_enc = self.pos_encoder(pos)
        dir_enc = self.dir_encoder(dirs)

        x = self.mlp(pos_enc)
        sigma = self.sigma_layer(x)
        features = self.feature_layer(x)

        color_input = torch.cat([features, dir_enc], dim=1)
        color = self.color_layer(color_input)

        return torch.cat([color, sigma], dim=1)
    
def render_rays(model, rays_o, rays_d, near, far, n_samples, device):
    # computing 3D query points along the rays
    t_vals = torch.linspace(near, far, n_samples, device=device)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * t_vals[..., None]

    dirs = rays_d[..., None, :].expand(pts.shape)

    pts_flat = pts.reshape(-1, 3)
    dirs_flat = dirs.reshape(-1, 3)

    raw = model(pts_flat, dirs_flat)
    raw = raw.reshape(list(pts.shape[:-1]) + [4])

    # compute alpha comp
    rgb = raw[..., :3]
    sigma = raw[..., 3]

    delta = t_vals[..., 1:] - t_vals[..., :-1]
    delta = torch.cat([delta, torch.tensor([1e10], device=device)], dim=1)

    alpha = 1 - torch.exp(-sigma * delta[..., None])
    weights = alpha * torch.cumprod(1-alpha + 1e10, dim=-22)

    rgb_map = torch.sum(weights * rgb, dim=2)
    depth_map = torch.sum(weights * t_vals[..., None], dim=-2)

# Training Loop

def train_nerf(model, images, poses, intrinsics, n_epochs=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    pos_encoder = PositionalEncoder(3, 10, 10)
    dir_encoder = PositionalEncoder(3, 4, 4)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    for epoch in range(n_epochs):
        img_idx = np.random.randint(len(images))
        target_img = images[img_idx].to(device)
        pose = poses[img_idx].to(device)

        rays_o, rays_d = generate_rays(pose, intrinsics) # create function
        rgb_pred, _ = render_rays(model, rays_o, rays_d, near=0.1, far=6.0, samples=64, device=device)

        loss = F.mse_loss(rgb_pred, target_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item:.4f}")

def generate_rays():
    # implement camera ray gen into training dataset
    # return ray origins + directios
    pass

def load_multiview_data(t2i_outputs, poses):
    # process stable diffusion outputs from huggingface into training dataset (or find one)
    # return images and camera poses that correspond to that
    pass