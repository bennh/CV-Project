# data_loader.py

import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to quaternion (w, x, y, z).
    """
    q = np.empty((4,), dtype=np.float32)
    tr = np.trace(R)

    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q[0] = (R[2, 1] - R[1, 2]) / S
            q[1] = 0.25 * S
            q[2] = (R[0, 1] + R[1, 0]) / S
            q[3] = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q[0] = (R[0, 2] - R[2, 0]) / S
            q[1] = (R[0, 1] + R[1, 0]) / S
            q[2] = 0.25 * S
            q[3] = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q[0] = (R[1, 0] - R[0, 1]) / S
            q[1] = (R[0, 2] + R[2, 0]) / S
            q[2] = (R[1, 2] + R[2, 1]) / S
            q[3] = 0.25 * S
    return q / np.linalg.norm(q)


def load_pose_file(pose_path):
    """
    Load a 4x4 pose matrix from .pose.txt file.
    Returns:
        T (ndarray): 4x4 pose matrix
        t (ndarray): translation (3,)
        q (ndarray): quaternion (4,) (w,x,y,z)
    """
    T = np.loadtxt(pose_path).astype(np.float32).reshape(4, 4)
    R, t = T[:3, :3], T[:3, 3]
    q = rotation_matrix_to_quaternion(R)
    return T, t, q


class SevenScenesDataset(Dataset):
    """
    7-Scenes Dataset Loader
    """
    def __init__(self, root, scene, split="train", img_size=(256, 320), return_full_pose=False):
        super().__init__()
        self.root = root
        self.scene = scene
        self.split = split.lower()
        self.img_size = img_size
        self.return_full_pose = return_full_pose

        # 根据 split 文件选择 sequence
        split_file = os.path.join(root, scene, f"{split.capitalize()}Split.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, "r") as f:
            seqs = [line.strip() for line in f]

        self.samples = []
        for seq in seqs:
            seq_dir = os.path.join(root, scene, seq)
            rgb_files = sorted(glob.glob(os.path.join(seq_dir, "*.color.png")))
            for rgb_path in rgb_files:
                base = rgb_path.replace(".color.png", "")
                depth_path = base + ".depth.png"
                pose_path = base + ".pose.txt"
                if os.path.exists(depth_path) and os.path.exists(pose_path):
                    self.samples.append((rgb_path, depth_path, pose_path))

        # 图像预处理
        self.to_tensor = T.Compose([
            T.ToPILImage(),
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, depth_path, pose_path = self.samples[idx]

        # RGB
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        img = self.to_tensor(rgb)

        # 位姿
        T, t, q = load_pose_file(pose_path)
        t = torch.from_numpy(t).float()
        q = torch.from_numpy(q).float()

        if self.return_full_pose:
            return img, t, q, torch.from_numpy(T).float()
        else:
            return img, t, q
