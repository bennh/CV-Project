import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


def matrix_to_quaternion(Rt):
    """
    将 4x4 pose 矩阵转为 (translation, quaternion)
    Args:
        Rt: [4,4] numpy array
    Returns:
        t: [3] translation
        q: [4] quaternion (x,y,z,w)
    """
    R = Rt[:3, :3]
    t = Rt[:3, 3]

    # 四元数 (x, y, z, w)
    qw = np.sqrt(1.0 + np.trace(R)) / 2.0
    qx = (R[2, 1] - R[1, 2]) / (4 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw)
    q = np.array([qx, qy, qz, qw])

    return t.astype(np.float32), q.astype(np.float32)


class SevenScenesDataset(Dataset):
    def __init__(self, root_dir, scene="chess", split="train", transform=None, return_depth=False):
        """
        PyTorch Dataset for 7-Scenes
        Args:
            root_dir: 数据集根目录 (e.g. '/path/to/7-scenes-dataset')
            scene: 场景名 (e.g. 'chess')
            split: 'train' / 'test' / 'all'
            transform: 图像预处理 (torchvision.transforms)
            return_depth: 是否返回深度图 (PnP/RANSAC baseline 用)
        """
        self.scene_path = os.path.join(root_dir, scene)
        self.transform = transform
        self.return_depth = return_depth

        # 读取 split 文件
        if split == "train":
            split_files = ["TrainSplit.txt"]
        elif split == "test":
            split_files = ["TestSplit.txt"]
        elif split == "all":
            split_files = ["TrainSplit.txt", "TestSplit.txt"]
        else:
            raise ValueError("split must be 'train', 'test' or 'all'")

        self.seq_list = []
        for sp in split_files:
            with open(os.path.join(self.scene_path, sp), "r") as f:
                self.seq_list.extend([line.strip() for line in f if line.strip()])

        # 收集样本路径
        self.samples = []
        for seq in self.seq_list:
            seq_dir = os.path.join(self.scene_path, seq)
            color_files = sorted([f for f in os.listdir(seq_dir) if f.endswith("color.png")])
            for fname in color_files:
                base = fname.replace(".color.png", "")
                self.samples.append({
                    "color": os.path.join(seq_dir, base + ".color.png"),
                    "depth": os.path.join(seq_dir, base + ".depth.png"),
                    "pose":  os.path.join(seq_dir, base + ".pose.txt"),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # ---- 图像 ----
        color = cv2.imread(sample["color"], cv2.IMREAD_COLOR)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)  # BGR → RGB

        if self.transform:
            color = self.transform(color)
        else:
            color = torch.from_numpy(color.transpose(2, 0, 1)).float() / 255.0

        # ---- 位姿 ----
        pose_mat = np.loadtxt(sample["pose"]).astype(np.float32)  # 4x4
        t, q = matrix_to_quaternion(pose_mat)

        pose_mat = torch.from_numpy(pose_mat)  # [4,4]
        t = torch.from_numpy(t)                # [3]
        q = torch.from_numpy(q)                # [4]

        # ---- 深度图 (可选) ----
        if self.return_depth:
            depth = cv2.imread(sample["depth"], cv2.IMREAD_UNCHANGED)
            depth = torch.from_numpy(depth.astype(np.float32))
            return {"image": color, "translation": t, "quaternion": q,
                    "pose_matrix": pose_mat, "depth": depth}
        else:
            return {"image": color, "translation": t, "quaternion": q,
                    "pose_matrix": pose_mat}

    def print_split_info(self):
        """打印当前场景的序列划分情况"""
        print(f"Scene: {os.path.basename(self.scene_path)}")
        print(f"Sequences: {', '.join(self.seq_list)}")
        print(f"Total samples: {len(self.samples)}")


# ---------------- Test ----------------
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import torchvision.transforms as T

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),  # PoseNet 论文用 224x224
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    dataset = SevenScenesDataset(root_dir="7-scenes-dataset", scene="chess",
                                 split="train", transform=transform, return_depth=True)
    dataset.print_split_info()

    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch in loader:
        print("Image batch:", batch["image"].shape)         # [B,3,224,224]
        print("Translation batch:", batch["translation"].shape)  # [B,3]
        print("Quaternion batch:", batch["quaternion"].shape)    # [B,4]
        print("Pose matrix batch:", batch["pose_matrix"].shape)  # [B,4,4]
        print("Depth batch:", batch["depth"].shape)         # [B,H,W]
        break
