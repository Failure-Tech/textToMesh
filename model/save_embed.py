# import torch
# import numpy as np
# from transformers import AutoModel, AutoTokenizer
# import os

# # Ensure the correct embeddings directory exists
# embeddings_folder = "./data/text_embeddings"
# os.makedirs(embeddings_folder, exist_ok=True)

# # Load LLaMA 3.1-8B with Cerebras (if available)
# model_name = "BAAI/bge-large-en-v1.5"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

# def get_embedding(text):
#     tokens = tokenizer(text, return_tensors="pt")
#     with torch.no_grad():
#         output = model(**tokens)
#     return output.last_hidden_state.mean(dim=1).squeeze().numpy()  # Mean pooling for fixed-size vector

# # Example: Generate embeddings for ModelNet40 categories
# categories = ["airplane", "bathtub", "bed", "bench"]  # Replace with full ModelNet40 classes

# # Save embeddings
# for category in categories:
#     embedding = get_embedding(f"A 3D model of a {category}")  # Custom prompt
#     np.save(os.path.join(embeddings_folder, f"{category}.npy"), embedding)

# print("Embeddings saved successfully!")

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
import os
import random

# Ensure the embeddings directory exists
embeddings_folder = "./data/embeddings"
os.makedirs(embeddings_folder, exist_ok=True)

# Load the embedding model
model_name = "BAAI/bge-large-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    tokens = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state.mean(dim=1).squeeze().numpy()  # Mean pooling for fixed-size vector

# ModelNet40 categories
categories = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car", "chair", "cone",
    "cup", "curtain", "desk", "door", "dresser", "flower pot", "glass box", "guitar", "keyboard",
    "lamp", "laptop", "mantel", "monitor", "nightstand", "person", "piano", "plant", "radio",
    "range hood", "sink", "sofa", "stairs", "stool", "table", "tent", "toilet", "tv stand",
    "vase", "wardrobe", "xbox"
]

# Variations per category
variations = {
    "airplane": ["toy airplane", "passenger airplane", "fighter jet", "private jet", "cargo plane"],
    "bed": ["king-sized bed", "queen-sized bed", "single bed", "bunk bed", "hospital bed"],
    "chair": ["wooden chair", "gaming chair", "office chair", "rocking chair", "foldable chair"],
    "car": ["sports car", "sedan", "pickup truck", "electric car", "classic car"],
    "sofa": ["leather sofa", "fabric couch", "sectional sofa", "recliner sofa", "loveseat"],
    "desk": ["office desk", "wooden desk", "gaming desk", "standing desk", "small study desk"],
    "bottle": ["plastic bottle", "glass bottle", "water bottle", "wine bottle", "sports bottle"],
}

# Generate embeddings
total_count = 0
for category in categories:
    prompts = variations.get(category, [f"A 3D model of a {category}"])  # Use variations if available
    while len(prompts) < 6:  # Ensure we get enough variations
        prompts.append(f"A realistic 3D model of a {category}")  # Generic fallback

    # Limit to ~6 variations per category
    for prompt in random.sample(prompts, min(6, len(prompts))):
        embedding = get_embedding(prompt)
        save_path = os.path.join(embeddings_folder, f"{category}_{total_count}.npy")
        np.save(save_path, embedding)
        total_count += 1

print(f"✅ {total_count} embeddings saved in: {embeddings_folder}")
