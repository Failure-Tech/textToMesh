import torch
import torch.nn as nn
import torch.optim as otpim
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DatasetLoader

# Load trained model
model.load_state_dict(torch.load("text_to_voxel_model.pth"))
model.eval()

# Test with a new text embedding
text_embedding = torch.randn(1, 4096)  # Replace with real embedding
generated_voxels = model(text_embedding).detach().numpy()

# Convert to mesh
verts, faces = voxel_to_mesh(generated_voxels[0, 0])

# Save as .obj file
import trimesh
mesh = trimesh.Trimesh(vertices=verts, faces=faces)
mesh.export("generated_model.obj")
