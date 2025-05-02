import torch.nn as nn
import torch

# Basic CNN impl
class CNN(nn.Module):
    def __init__(self, token_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=2, padding=1), # 64 x H/2 x W/2
            nn.ReLU(), # non linearity introduction
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 128 x H/4 x W/4
            nn.ReLU(),
            nn.Conv2d(128, token_dim, kernel_size=4, stride=2, padding=1), # 256 x H/8 x W/8
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)
    
class ViewPointEmbedding(nn.Module):
    def __init__(self, num_views=16, token_dim=256):
        super().__init__()
        self.view_embed = nn.Embedding(num_views, token_dim) # tokenizing image

    def forward(self, view_id):
        return self.view_embed(view_id)
    
# Combine image + view tokens
