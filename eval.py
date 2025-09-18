# eval.py

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd

from data_loader import SevenScenesDataset
from models import PoseNet
from utils import load_ckpt, pose_err_trans_m, pose_err_angular_deg, to_numpy


def evaluate(args):
    # 数据集 (测试集)
    ds = SevenScenesDataset(args.data_root, args.scene, split="test")
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    # 模型
    model = PoseNet("resnet34").cuda().eval()
    load_ckpt(model, args.ckpt)

    # 结果
    trans_errs, rot_errs = [], []

    with torch.no_grad():
        for img, t_gt, q_gt in dl:
            img = img.cuda()
            pred = model(img)[0].cpu()

            t_pred, q_pred = pred[:3], pred[3:]
            t_gt, q_gt = t_gt[0], q_gt[0]

            t_err = pose_err_trans_m(to_numpy(t_pred), to_numpy(t_gt))
            r_err = pose_err_angular_deg(to_numpy(q_pred), to_numpy(q_gt))

            trans_errs.append(t_err)
            rot_errs.append(r_err)

    trans_errs = np.array(trans_errs)
    rot_errs = np.array(rot_errs)

    results = {
        "scene": args.scene,
        "mean_t": trans_errs.mean(),
        "median_t": np.median(trans_errs),
        "mean_r": rot_errs.mean(),
        "median_r": np.median(rot_errs),
    }

    print(f"[{args.scene}] "
          f"mean_t = {results['mean_t']:.3f} m | median_t = {results['median_t']:.3f} m || "
          f"mean_r = {results['mean_r']:.2f}° | median_r = {results['median_r']:.2f}°")

    # 保存到 CSV
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame([results]).to_csv(args.out, index=False)
    print(f"Results saved to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of 7Scenes dataset")
    parser.add_argument("--scene", type=str, required=True,
                        help="Scene to evaluate (e.g., chess, pumpkin, redkitchen)")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to trained checkpoint (best.ckpt)")
    parser.add_argument("--out", type=str, default="results/eval_results.csv",
                        help="Path to save evaluation results")
    args = parser.parse_args()

    evaluate(args)
