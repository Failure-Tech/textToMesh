import os
import shutil
from gradio_client import Client

# Prompt input
prompt = input("What do you want? ")

# Initialize client
client = Client("stabilityai/stable-diffusion")

result = client.predict(
    prompt=prompt,
    negative="create 3 viewpoints based off the given prompt",
    scale=9,
    api_name="/infer"
)

output_dir = "images_test"
os.makedirs(output_dir, exist_ok=True)

# Copy images from temp path to output directory
for i, item in enumerate(result):
    src_path = item["image"]
    filename = f"output_{i+1}.jpg"
    dst_path = os.path.join(output_dir, filename)
    shutil.copy(src_path, dst_path)
    print(f"Saved: {dst_path}")
