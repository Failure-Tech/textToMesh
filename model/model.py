import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# class TextToVoxelDataset(Dataset):
#     def __init__(self, embeddings_folder, voxels_folder):
#         self.embeddings_files = sorted(os.listdir(embeddings_folder))[:5]
#         self.voxel_files = sorted(os.listdir(voxels_folder))

#         # assert len(self.embeddings_files) == len(self.voxels_files), "Mismatch in embeddings and voxels"
#         assert len(self.embeddings_files) == len(self.voxel_files), (
#             f"Mismatch in embeddings ({len(self.embeddings_files)}) and voxels ({len(self.voxel_files)})"
#         )
#         self.embeddings_folder = embeddings_folder
#         self.voxels_folder = voxels_folder

#     def __len__(self):
#         return len(self.embeddings_files)

#     def __getitem__(self, idx):
#         embedding_path = os.path.join(self.embeddings_folder, self.embeddings_files[idx])
#         voxel_path = os.path.join(self.voxels_folder, self.voxel_files[idx])

#         embedding = np.load(embedding_path)
#         voxel = np.load(voxel_path)

#         return (torch.tensor(embedding, dtype=torch.float32), torch.tensor(voxel, dtype=torch.float32))

class TextToVoxelDataset(Dataset):
    def __init__(self, embeddings_folder, voxels_folder):
        self.embeddings_files = sorted(os.listdir(embeddings_folder))[:5]  # Only use first 5 embeddings

        # Recursively find all .npy files inside train/ and test/ subfolders
        self.voxel_files = sorted([
            os.path.join(root, file)
            for category in os.listdir(voxels_folder)  # e.g., 'airplane', 'car', etc.
            for root, _, files in os.walk(os.path.join(voxels_folder, category))  # Traverse train/ and test/
            for file in files if file.endswith(".npy")
        ])[:5]  # Limit to 5 voxel files

        assert len(self.embeddings_files) == len(self.voxel_files), (
            f"Mismatch in embeddings ({len(self.embeddings_files)}) and voxels ({len(self.voxel_files)})"
        )

        self.embeddings_folder = embeddings_folder
        self.voxels_folder = voxels_folder

    def __len__(self):
        return len(self.embeddings_files)

    def __getitem__(self, idx):
        embedding_path = os.path.join(self.embeddings_folder, self.embeddings_files[idx])
        voxel_path = self.voxel_files[idx]  # Direct path from the recursive search

        embedding = np.load(embedding_path)
        voxel = np.load(voxel_path)

        return torch.tensor(embedding, dtype=torch.float32), torch.tensor(voxel, dtype=torch.float32)

embeddings_folder = "./data/embeddings" # custom generated so may not be reliable
voxels_folder = "./data/ModelNet40_voxels"
dataset = TextToVoxelDataset(embeddings_folder, voxels_folder)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
# validation_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# class TextToVoxel(nn.Module):
#     def __init__(self, embedding_dim=1024, voxel_size=64):
#         super(TextToVoxel, self).__init__()

#         self.fc1 = nn.Linear(embedding_dim, 512)
#         self.fc2 = nn.Linear(512, 8*8*8)  # Latent space (8³ grid)
        
#         self.deconv1 = nn.ConvTranspose3d(1, 64, kernel_size=4, stride=2, padding=1)
#         self.deconv2 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1)
#         self.deconv3 = nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1)

#     def forward(self, text_embedding):
#         x = F.relu(self.fc1(text_embedding))
#         x = F.relu(self.fc2(x))
#         x = x.view(-1, 1, 8, 8, 8)  # Reshape into 3D space
        
#         x = F.relu(self.deconv1(x))
#         x = F.relu(self.deconv2(x))
#         print(f"Values before squash: {x}")
#         x = torch.sigmoid(self.deconv3(x))  # Output voxel grid
#         print(f"Values after squash: {x}")
#         return x

class TextToVoxel(nn.Module):
    def __init__(self, embedding_dim=1024, voxel_size=64):
        super(TextToVoxel, self).__init__()
        
        # Fully connected layers to process the text embedding
        self.fc1 = nn.Linear(embedding_dim, 2048)  # Larger hidden layer
        self.fc2 = nn.Linear(2048, 1024)  # Reducing to a more manageable size
        
        # Latent space reshaping
        self.fc3 = nn.Linear(1024, 8 * 8 * 8)  # Start with a latent space 8x8x8

        # Decoder part with larger voxel output (64x64x64)
        self.deconv1 = nn.ConvTranspose3d(1, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1)
        
        # Optional: Using skip connections
        self.skip1 = nn.ConvTranspose3d(1, 256, kernel_size=4, stride=2, padding=1)
        self.skip2 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)

    def forward(self, text_embedding):
        # Process the text embedding through fully connected layers
        x = F.relu(self.fc1(text_embedding))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view(-1, 1, 8, 8, 8)  # Reshaping the output to a latent space (8x8x8)

        # Decoder with skip connections
        x = F.relu(self.deconv1(x))
        skip1 = self.skip1(x)
        x = F.relu(self.deconv2(x))
        skip2 = self.skip2(x)
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = torch.sigmoid(self.deconv5(x))  # Final output voxel grid

        return x

model = TextToVoxel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
epoch_losses = []

for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (embeddings, voxels) in enumerate(dataloader):
        optimizer.zero_grad()
        voxels = voxels.unsqueeze(1)
        predicted_voxels = model(embeddings)
        loss = criterion(predicted_voxels, voxels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    epoch_losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")

torch.save(model.state_dict(), "text_to_voxel_model.pth")

plt.plot(range(1, num_epochs+1), epoch_losses, marker="o", color="b", label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.legend()
plt.show()

text_embedding = torch.randn(1, 1024)  # Example text embedding
model = TextToVoxel()
model.load_state_dict(torch.load("text_to_voxel_model.pth"))

model.eval()
# with torch.no_grad():
#     total_loss = 0
#     for embeddings, voxels in validation_dataloder:
#         predicted_voxels = model(embeddings)
#         loss = criterion(predicted_voxels, voxels)
#         total_loss += loss.item()
#     avg_loss = total_loss / len(validation_dataloder)
#     print(f"Validation Loss: {avg_loss:.4f}")

voxel_output = model(text_embedding)
print(voxel_output.shape)  # Should output (1, 1, 64, 64, 64)
