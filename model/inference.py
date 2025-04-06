# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from model import TextToVoxel  # Assuming your model is in a file named model.py
# import trimesh
# from transformers import BertTokenizer, BertModel

# # Step 1: Load the trained model
# model = TextToVoxel()
# model.load_state_dict(torch.load("text_to_voxel_model.pth"))
# model.eval()  # Set the model to evaluation mode

# # Step 2: Provide an input text embedding
# # You can use a random tensor or load a specific embedding
# # text_embedding = torch.randn(1, 1024)  # Example text embedding (random for now)
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# bert_model = BertModel.from_pretrained("bert-base-uncased")

# def get_text_embedding(text):
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#     outputs = bert_model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1)  # Averaging the hidden states as a simple embedding

# # Example text input
# text_input = "A 3D model of a airplane"

# # Get the embedding from the text input
# text_embedding = get_text_embedding(text_input)

# # Step 3: Get the output from the model
# # with torch.no_grad():  # No need to compute gradients during inference
# #     voxel_output = model(text_embedding)
# voxel_output = model(text_embedding)

# # Step 4: Visualize the 3D voxel grid
# voxel_output = voxel_output.squeeze().numpy()  # Remove the batch dimension and convert to numpy array

# # Create a 3D plot of the voxel output
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.voxels(voxel_output > 0.5, edgecolors="k")  # Visualize where the voxel is non-zero

# ax.set_title("Generated 3D Voxel Grid")
# plt.show()

# mesh = trimesh.voxel.ops.matrix_to_marching_cubes(voxel_output > 0.5)
# mesh.show()
# mesh.export("generated_mesh.obj")

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from model import TextToVoxel  # Assuming your model is in model.py
import trimesh
from transformers import BertTokenizer, BertModel

# Step 1: Load the trained model
model = TextToVoxel()
model.load_state_dict(torch.load("text_to_voxel_model.pth"))
model.eval()  # Set the model to evaluation mode

# Step 2: Provide an input text embedding
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Averaging the hidden states as a simple embedding

# Example text input
text_input = "A model of a bed"

# Get the embedding from the text input
text_embedding = get_text_embedding(text_input)

# Resize the embedding to match the expected input size (1024)
embedding_resized = torch.nn.functional.interpolate(text_embedding.unsqueeze(1), size=(1024,), mode='linear').squeeze()

# Step 3: Get the output from the model
voxel_output = model(embedding_resized)

# Step 4: Visualize the 3D voxel grid
voxel_output = voxel_output.squeeze().detach().numpy()  # Remove the batch dimension and convert to numpy array

# Create a 3D plot of the voxel output
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.voxels(voxel_output > 0.5, edgecolors="k")  # Visualize where the voxel is non-zero

ax.set_title("Generated 3D Voxel Grid")
plt.show()

mesh = trimesh.voxel.ops.matrix_to_marching_cubes(voxel_output > 0.5)
mesh.show()
mesh.export("generated_mesh.obj")
