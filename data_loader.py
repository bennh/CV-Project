import os
import numpy as np
import cv2

class SevenScenesDataset:
    def __init__(self, root_dir, scene="chess", split="train"):
        """
        root_dir: 数据集根目录 (e.g. '/path/to/7-scenes-dataset')
        scene: 场景名 (e.g. 'chess')
        split: 'train' / 'test' / 'all'
        """
        self.scene_path = os.path.join(root_dir, scene)

        if split == "train":
            split_file = "TrainSplit.txt"
            split_path = os.path.join(self.scene_path, split_file)
            with open(split_path, "r") as f:
                self.seq_list = [line.strip() for line in f if line.strip()]

        elif split == "test":
            split_file = "TestSplit.txt"
            split_path = os.path.join(self.scene_path, split_file)
            with open(split_path, "r") as f:
                self.seq_list = [line.strip() for line in f if line.strip()]

        elif split == "all":
            train_file = os.path.join(self.scene_path, "TrainSplit.txt")
            test_file  = os.path.join(self.scene_path, "TestSplit.txt")
            self.seq_list = []
            for sp in [train_file, test_file]:
                with open(sp, "r") as f:
                    self.seq_list.extend([line.strip() for line in f if line.strip()])
        else:
            raise ValueError("split must be 'train', 'test' or 'all'")

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
        print(f"Sequences: {', '.join(self.seq_list)}")
        print(f"Total samples: {len(self.samples)}")


if __name__ == "__main__":
    # 示例 1: 只加载 train
    train_set = SevenScenesDataset(root_dir="7-scenes-dataset", scene="chess", split="train")
    train_set.print_split_info()

    # 示例 2: 只加载 test
    test_set = SevenScenesDataset(root_dir="7-scenes-dataset", scene="chess", split="test")
    test_set.print_split_info()

    # 示例 3: train+test 全部加载
    all_set = SevenScenesDataset(root_dir="7-scenes-dataset", scene="chess", split="all")
    all_set.print_split_info()
