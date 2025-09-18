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
    # 数据集
    ds = SevenScenesDataset(args.data_root, args.scene, split="test")
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    # 模型
    model = PoseNet("resnet34").cuda().eval()
    load_ckpt(model, args.ckpt)

    # 逐帧结果
    results = []

    with torch.no_grad():
        for idx, (img, t_gt, q_gt) in enumerate(dl):
            img = img.cuda()
            pred = model(img)[0].cpu()

            t_pred, q_pred = pred[:3], pred[3:]
            t_gt, q_gt = t_gt[0], q_gt[0]

            t_err = pose_err_trans_m(to_numpy(t_pred), to_numpy(t_gt))
            r_err = pose_err_angular_deg(to_numpy(q_pred), to_numpy(q_gt))

            results.append({
                "frame": idx,
                "t_err_m": t_err,
                "r_err_deg": r_err,
                "t_gt_x": float(t_gt[0]), "t_gt_y": float(t_gt[1]), "t_gt_z": float(t_gt[2]),
                "t_pred_x": float(t_pred[0]), "t_pred_y": float(t_pred[1]), "t_pred_z": float(t_pred[2]),
            })

    df = pd.DataFrame(results)

    # 统计
    stats = {
        "scene": args.scene,
        "mean_t": df["t_err_m"].mean(),
        "median_t": df["t_err_m"].median(),
        "mean_r": df["r_err_deg"].mean(),
        "median_r": df["r_err_deg"].median()
    }

    print(f"[PoseNet | {args.scene}] "
          f"mean_t={stats['mean_t']:.3f} m | median_t={stats['median_t']:.3f} m || "
          f"mean_r={stats['mean_r']:.2f}° | median_r={stats['median_r']:.2f}°")

    # 保存逐帧和统计
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    per_frame_csv = args.out.replace(".csv", "_perframe.csv")
    df.to_csv(per_frame_csv, index=False)
    pd.DataFrame([stats]).to_csv(args.out, index=False)

    return stats, df
