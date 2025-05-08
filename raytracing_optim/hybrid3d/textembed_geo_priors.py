from transformers import CLIPTokenizer, CLIPTextModel
import torch

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
user_input = input("What do you want? ")

prompt = user_input
inputs = tokenizer(prompt, return_tensors="pt")
text_embed = text_encoder(**inputs).last_hidden_state.mean(dim=1)

view_dirs = torch.tensor([
    [1.0, 0.0, 0.0], # front
    [0.0, 1.0, 0.0], # side
    [0.0, 0.0, 1.0] # top
], dytpe=torch.float32)

embed_w_views = []
for dir_vec in view_dirs:
    combined = torch.cat([text_embed.squeeze(), dir_vec], dim=0)
    embed_w_views.append(combined)

embed_w_views = torch.stack(embed_w_views) # [3, 515]

feature_dim = 64
voxel_grid = torch.rand(64, 64, 64, feature_dim, requires_grad=True)

def simple_raytrace(voxel_grid, direction, embedding):
    traced_voxels = []
    for t in torch.linspace(0, 1, steps=32):
        point = t*direction
        x, y, z = ((point + 1) * 31.5).long()
        x, y, z = torch.clamp((torch.tensor[x, y, z]), 0, 63)
        voxel = voxel_grid[x, y, z]
        mixed = voxel + embedding[:feature_dim]
        traced_voxels.append(mixed)

    return torch.stack(traced_voxels).mean(dim=0)

features_per_view = []
for i in range(3): # for 3 view points
    ray_feature = simple_raytrace(voxel_grid, view_dirs[i], embed_w_views[i])
    features_per_view.append(ray_feature)

final = torch.stack(features_per_view).mean(dim=0)