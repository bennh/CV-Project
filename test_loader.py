# test_loader.py

import torch
from data_loader import SevenScenesDataset

def main():
    data_root = "/path/to/7Scenes"  # Change to your dataset path
    scene = "chess"

    # Load the training set and return the full pose matrix
    ds = SevenScenesDataset(data_root, scene, split="train", return_full_pose=True)

    print(f"Loaded {scene} train set: {len(ds)} samples")

    # Take the first sample
    img, t, q, T = ds[0]

    print("Image tensor shape:", img.shape)   # (3, H, W)
    print("Translation vector t:", t)
    print("Quaternion q (w,x,y,z):", q)
    print("Full 4x4 pose matrix T:\n", T)

if __name__ == "__main__":
    main()
