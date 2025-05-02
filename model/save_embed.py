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

# Set random seed for reproducibility
random.seed(42)

# Ensure the embeddings directory exists
embeddings_folder = "./data/embeddings"
os.makedirs(embeddings_folder, exist_ok=True)

# Load the embedding model
model_name = "BAAI/bge-large-en-v1.5" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state.mean(dim=1).squeeze().numpy()

# Categories and their variations
target_categories = ["airplane", "bathtub", "bench", "bed", "bookshelf"]
variations = {
    "airplane": ["toy airplane", "passenger airplane", "fighter jet", "private jet", "cargo plane"],
    "bed": ["king-sized bed", "queen-sized bed", "single bed", "bunk bed", "hospital bed"],
    "bench": ["park bench", "wooden bench", "metal bench", "indoor bench", "gym bench"],
    "bathtub": ["clawfoot bathtub", "jacuzzi bathtub", "modern bathtub", "round bathtub", "freestanding tub"],
    "bookshelf": ["wooden bookshelf", "metal bookshelf", "tall bookshelf", "wall-mounted bookshelf", "corner bookshelf"],
}

# Determine how many embeddings per category
total_embeddings = 1839
base_count = total_embeddings // len(target_categories)  # 367
extras = total_embeddings % len(target_categories)       # 4

# Assign extra embeddings to the first few categories
category_counts = {cat: base_count + (1 if i < extras else 0) for i, cat in enumerate(target_categories)}

# Generator function for prompts
def generate_prompts(category, count):
    base_prompts = variations.get(category, [f"A 3D model of a {category}"])
    prompts = []
    while len(prompts) < count:
        base = random.choice(base_prompts)
        # Add optional adjectives or forms to diversify
        adjectives = ["realistic", "polygonal", "low-poly", "high-detail", "rendered", "stylized", "wireframe"]
        style = random.choice(adjectives)
        prompts.append(f"A {style} {base}")
    return prompts

# Generate and save embeddings
total_count = 0
for category in target_categories:
    count = category_counts[category]
    prompts = generate_prompts(category, count)
    for i, prompt in enumerate(prompts):
        embedding = get_embedding(prompt)
        save_path = os.path.join(embeddings_folder, f"{category}_{i}.npy")
        np.save(save_path, embedding)
        total_count += 1

print(f"✅ {total_count} embeddings saved for categories: {target_categories}")

