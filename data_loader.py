import os
import numpy as np
import cv2

class SevenScenesDataset:
    def __init__(self, root_dir, scene="chess", split="train"):
        """
        root_dir: 数据集根目录 (e.g. '/path/to/7-scenes-dataset')
        scene: 场景名 (e.g. 'chess')
        split: 'train' or 'test'
        """
        self.scene_path = os.path.join(root_dir, scene)
        split_file = "TrainSplit.txt" if split == "train" else "TestSplit.txt"
        split_path = os.path.join(self.scene_path, split_file)

        # 读取 train/test split 文件
        with open(split_path, "r") as f:
            self.seq_list = [line.strip() for line in f if line.strip()]

        # 收集所有样本路径
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

        color = cv2.imread(sample["color"], cv2.IMREAD_COLOR)
        depth = cv2.imread(sample["depth"], cv2.IMREAD_UNCHANGED)
        pose = np.loadtxt(sample["pose"])

        return color, depth, pose

    def print_split_info(self):
        """打印当前场景的序列划分情况"""
        print(f"Scene: {os.path.basename(self.scene_path)}")
        print(f"Split type: {len(self.seq_list)} sequences ({', '.join(self.seq_list)})")
        print(f"Total samples: {len(self.samples)}")


if __name__ == "__main__":
    dataset = SevenScenesDataset(root_dir="7-scenes-dataset", scene="chess", split="train")
    dataset.print_split_info()

    color, depth, pose = dataset[0]
    print("Color image shape:", color.shape)
    print("Depth image shape:", depth.shape)
    print("Pose matrix:\n", pose)
