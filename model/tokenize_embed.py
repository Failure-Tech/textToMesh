# from cerebras.cloud.sdk import Cerebras
# import os

# # api_key = os.environ.get("CEREBRAS_API_KEY")
# api_key = "csk-9hxpm328d445wpyhvj8expnhmrpvcp8rh6d32tjmvfww8n8h"
# prompt = input("Enter your prompt: ")
# print(f"{prompt}")

# client = Cerebras(
#     api_key=api_key
# )

# def get_text_embedding(prompt):
#     try:
#         response = client.chat.completions.create(
#             messages=[{
#                 "role": "user",
#                 "content": prompt
#             }],
#             model="llama3.1-8b",
#         )
#         embedding = response.choices[0].
#         print(f"Embedding: {embedding}")
#         return embedding

#         # print(response.choices[0].message.content)
#         # return response.choices[0].message.content
#     except Exception as e:
#         print(f"Error in Cerebras preprocessing: {e}")

# embedding_vector = get_text_embedding(prompt)

from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

# python3 -m pip install --upgrade 'optree>=0.13.0

if not hasattr(np, 'object'):
    np.object = object

# Load LLaMA 3.1-8B model and tokenizer
model_name = "BAAI/bge-large-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use the last hidden state mean as the embedding
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return embedding

prompt = input("Enter your prompt: ")
embedding_vector = get_text_embedding(prompt)
print(f"Embedding (first 5 values): {embedding_vector[:5]}...")