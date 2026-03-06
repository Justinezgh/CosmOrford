#!/usr/bin/env python3
"""Download dataset, unzip, and build HuggingFace dataset."""
import argparse
import subprocess
import zipfile
from pathlib import Path
import numpy as np
from datasets import Dataset, concatenate_datasets, Features, Array2D, Sequence, Value
from tqdm import tqdm
from cosmoford.utils import Data


parser = argparse.ArgumentParser(description='Build and upload HF dataset')
parser.add_argument('--repo', default='cosmostat/neurips-wl-challenge-flat', help='HuggingFace repo name')
parser.add_argument('--url', default='https://storage.googleapis.com/neurips-wl/public_data.zip', help='Dataset URL')
parser.add_argument('--data-dir', default='data', help='Data directory')
args = parser.parse_args()

data_dir = Path(args.data_dir)
zip_path = data_dir / "dataset.zip"

try:
    # Load data
    print("Loading data...")
    data_obj = Data(data_dir=str(data_dir), USE_PUBLIC_DATASET=True)
    data_obj.load_train_data()

except FileNotFoundError:
    
    # Download
    if zip_path.exists():
        print(f"✓ Dataset zip already exists at {zip_path}, skipping download")
    else:
        print(f"Downloading from {args.url}...")
        data_dir.mkdir(parents=True, exist_ok=True)
        # Try wget first, fallback to curl
        try:
            subprocess.run(['wget', '-O', str(zip_path), args.url], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            subprocess.run(['curl', '-L', '-o', str(zip_path), args.url], check=True)
        print(f"✓ Downloaded to {zip_path}")

    print("Unzipping...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print(f"✓ Extracted to {data_dir}")

    # Load data
    print("Loading data...")
    data_obj = Data(data_dir=str(data_dir), USE_PUBLIC_DATASET=True)
    data_obj.load_train_data()

# Build HF dataset
# transposing the nuisance parameters to axis 0 to easily split along its axis
# this reshapes the dataset such as
# ncosmo, np, ... -> np, ncosmo, ... 
print("Building HF dataset...")

dsets = []
for i in tqdm(range(data_obj.label.shape[1])):
    dset = Dataset.from_dict(
        {'kappa': data_obj.kappa[:,i], 'theta': data_obj.label[:,i]},
        features=Features({
            "kappa": Array2D(shape=(1424, 176), dtype="float16"),
            "theta": Sequence(Value("float32"), length=5)
        })
    )
    dsets.append(dset)

# Apply random permutation of datasets to mix nuisance parameters, but with
# fixed seed for reproducibility
rng = np.random.default_rng(seed=42)
perm = rng.permutation(len(dsets))
dsets = [dsets[i] for i in perm]

dset_train = concatenate_datasets(dsets[:200], split='train')
dset_val = concatenate_datasets(dsets[200:], split='validation')

# Push to HF
print(f"Pushing to {args.repo}...")
dset_train.push_to_hub(args.repo, split='train')
dset_val.push_to_hub(args.repo, split='validation')
print("Done!")
