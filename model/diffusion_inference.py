import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the same model architecture for loading
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

class UNet3D(nn.Module):
    def __init__(self, cond_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(2, 32, 3, padding=1),
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
        batch_size = x.shape[0]
        t_channel = t_emb[:, :1].view(batch_size, 1, 1, 1, 1).expand(-1, -1, *x.shape[2:])
        x = torch.cat([x, t_channel], dim=1)
        
        x = self.encoder(x)
        cond = cond.view(cond.size(0), cond.size(1), 1, 1, 1).expand(-1, -1, *x.shape[2:])
        x = torch.cat([x, cond], dim=1)
        x = self.middle(x)
        x = self.decoder(x)
        return x

# Sinusoidal timestep embedding
def get_timestep_embedding(timesteps, dim):
    half_dim = dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

# Text-to-embedding function (you'd normally use a pretrained model like CLIP)
def generate_text_embedding(text, embedding_dim=1024):
    """
    For demonstration purposes, this generates a random embedding.
    In a real application, you would use a pretrained text encoder.
    """
    # Use a seed based on the text for reproducible results
    seed = sum(ord(c) for c in text)
    np.random.seed(seed)
    
    # Generate a random embedding with specific characteristics based on the text
    embedding = np.random.randn(embedding_dim)
    
    # Normalize the embedding
    embedding = embedding / np.linalg.norm(embedding)
    
    return torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

# Diffusion inference model
class DiffusionInference:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load the saved model
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']
        
        # Initialize models with the same architecture
        self.projector = TextConditionProjector(
            embedding_dim=config['embedding_dim'], 
            latent_dim=config['latent_dim']
        ).to(self.device)
        self.unet = UNet3D(cond_dim=config['latent_dim']).to(self.device)
        
        # Load state dictionaries
        self.projector.load_state_dict(checkpoint['projector_state_dict'])
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        
        # Load diffusion parameters
        self.timesteps = config['timesteps']
        self.betas = checkpoint['betas'].to(self.device)
        self.alphas = checkpoint['alphas'].to(self.device)
        self.alpha_cumprod = checkpoint['alpha_cumprod'].to(self.device)
        
        print(f"Model loaded successfully from {model_path}")
    
    def generate_voxel(self, text, resolution=64, steps=None):
        """Generate a voxel model from text description"""
        if steps is None:
            steps = self.timesteps
            
        # Set models to evaluation mode
        self.projector.eval()
        self.unet.eval()
        
        # Generate text embedding
        text_embedding = generate_text_embedding(text).to(self.device)
        
        # Project text embedding to conditioning space
        cond = self.projector(text_embedding)
        
        # Start with random noise
        x = torch.randn((1, 1, resolution, resolution, resolution), device=self.device)
        
        print(f"Generating voxel model for text: '{text}'")
        print(f"Running diffusion sampling for {steps} steps...")
        
        # Sampling loop
        with torch.no_grad():
            for i, t in enumerate(reversed(range(self.timesteps))):
                if i % (self.timesteps // 10) == 0:
                    print(f"Sampling step {i}/{self.timesteps}")
                    
                t_tensor = torch.tensor([t], dtype=torch.long, device=self.device)
                t_emb = get_timestep_embedding(t_tensor, cond.size(1))
                
                # Predict noise
                eps = self.unet(x, t_emb, cond)
                
                # Update sample
                alpha = self.alphas[t]
                alpha_bar = self.alpha_cumprod[t]
                beta = self.betas[t]
                
                # Perform the update step
                x = (1 / alpha**0.5) * (x - (1 - alpha)**0.5 * eps)
                
                # Add noise if not the last step
                if t > 0:
                    noise = torch.randn_like(x)
                    x += beta**0.5 * noise
        
        # Convert to binary voxels with threshold
        voxel = x.squeeze().cpu().numpy()
        return voxel

    def visualize_voxel(self, voxel, threshold=0.5, output_file=None):
        """Visualize the generated voxel model"""
        # Threshold the voxel data
        binary_voxel = voxel > threshold
        
        # Get the coordinates of filled voxels
        filled_voxels = np.where(binary_voxel)
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot filled voxels as scattered points
        ax.scatter(filled_voxels[0], filled_voxels[1], filled_voxels[2], 
                  s=5, c='blue', alpha=0.5)
        
        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Set title
        ax.set_title('Generated Voxel Model')
        
        # Save or show the figure
        if output_file:
            plt.savefig(output_file)
            print(f"Visualization saved to {output_file}")
        else:
            plt.show()
        
        plt.close()

    def save_voxel(self, voxel, output_file):
        """Save the voxel model to a NumPy file"""
        np.save(output_file, voxel)
        print(f"Voxel model saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate a 3D voxel model from text description')
    parser.add_argument('--model_path', type=str, default='text_to_voxel_model.pt', 
                        help='Path to the saved diffusion model')
    parser.add_argument('--text', type=str, required=True, 
                        help='Text description of the 3D object to generate')
    parser.add_argument('--output', type=str, default='generated_voxel.npy',
                        help='Output path for the generated voxel model')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the generated voxel model')
    parser.add_argument('--vis_output', type=str, default=None,
                        help='Path to save the visualization image')
    parser.add_argument('--steps', type=int, default=None,
                        help='Number of diffusion steps for sampling (default: use model default)')
    
    args = parser.parse_args()
    
    # Initialize inference model
    inference = DiffusionInference(args.model_path)
    
    # Generate voxel model
    voxel = inference.generate_voxel(args.text, steps=args.steps)
    
    # Save the generated voxel model
    inference.save_voxel(voxel, args.output)
    
    # Visualize if requested
    if args.visualize:
        inference.visualize_voxel(voxel, output_file=args.vis_output)

if __name__ == "__main__":
    main()