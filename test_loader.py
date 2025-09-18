# test_loader.py

import torch
from data_loader import SevenScenesDataset

def main():
    data_root = "/path/to/7Scenes"  # 修改成你的数据集路径
    scene = "chess"

    # 加载训练集，返回完整位姿矩阵
    ds = SevenScenesDataset(data_root, scene, split="train", return_full_pose=True)

    print(f"Loaded {scene} train set: {len(ds)} samples")

    # 取第一张
    img, t, q, T = ds[0]

    print("Image tensor shape:", img.shape)   # (3, H, W)
    print("Translation vector t:", t)
    print("Quaternion q (w,x,y,z):", q)
    print("Full 4x4 pose matrix T:\n", T)

if __name__ == "__main__":
    main()
