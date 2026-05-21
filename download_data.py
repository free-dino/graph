import kagglehub
import shutil
from pathlib import Path

# Download dataset
path = kagglehub.dataset_download("minhhhtrann/graph-shopee")

print("Downloaded to:", path)

# Current directory
current_dir = Path.cwd()

# Copy all files into current directory
src = Path(path)

for item in src.iterdir():
    dest = current_dir / item.name

    if item.is_dir():
        shutil.copytree(item, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(item, dest)

print("Dataset copied to:", current_dir)