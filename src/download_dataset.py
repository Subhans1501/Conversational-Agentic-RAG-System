import os
import shutil
from huggingface_hub import snapshot_download

target_dir = "data/raw"
os.makedirs(target_dir, exist_ok=True)

print(f"Downloading Daily Dialog dataset to '{target_dir}'...")
snapshot_download(
    repo_id="DeepPavlov/daily_dialog",
    repo_type="dataset",
    local_dir=target_dir,
    allow_patterns=["data/*"] 
)

nested_data_dir = os.path.join(target_dir, "data")

if os.path.exists(nested_data_dir):
    print("Moving files out of the nested folder...")
    for item in os.listdir(nested_data_dir):
        source_path = os.path.join(nested_data_dir, item)
        destination_path = os.path.join(target_dir, item)
        shutil.move(source_path, destination_path)
    
    os.rmdir(nested_data_dir)

print("Download complete!")