# utils.py

import os
import torch
import torch.nn as nn
import numpy as np
import random


# ====================
# 损失函数
# ====================
class PoseLoss(nn.Module):
    """
    PoseNet 损失函数:
    L = ||t - t_gt||_2 + beta * ||q - q_gt||_2
    """
    def __init__(self, beta=120.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred, t_gt, q_gt):
        t = pred[:, :3]
        q = pred[:, 3:]
        q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
        t_loss = torch.norm(t - t_gt, dim=1).mean()
        q_loss = torch.norm(q - q_gt, dim=1).mean()
        return t_loss + self.beta * q_loss


# ====================
# 随机种子
# ====================
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ====================
# 模型保存 / 加载
# ====================
def save_ckpt(model, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, name))


def load_ckpt(model, path):
    sd = torch.load(path, map_location="cpu")
    model.load_state_dict(sd)


# ====================
# 姿态误差计算
# ====================
def pose_err_trans_m(t_pred, t_gt):
    """ 平移误差 (米) """
    return float(np.linalg.norm(t_pred - t_gt))


def pose_err_angular_deg(q_pred, q_gt):
    """ 四元数旋转误差 (角度) """
    q_pred = q_pred / (np.linalg.norm(q_pred) + 1e-12)
    q_gt = q_gt / (np.linalg.norm(q_gt) + 1e-12)
    d = abs(np.dot(q_pred, q_gt))
    d = np.clip(d, -1.0, 1.0)
    return float(2 * np.degrees(np.arccos(d)))


# ====================
# 四元数 / SE3 转换
# ====================
def quat2mat(q):
    """ 四元数 -> 旋转矩阵 """
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y*y + z*z),     2 * (x*y - z*w),     2 * (x*z + y*w)],
        [    2 * (x*y + z*w), 1 - 2 * (x*x + z*z),     2 * (y*z - x*w)],
        [    2 * (x*z - y*w),     2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]
    ], dtype=np.float32)
    return R


def se3_to_tq(T):
    """ SE3 4x4 -> 平移 + 四元数 """
    t = T[:3, 3]
    R = T[:3, :3]
    q = rotmat_to_quat(R)
    return t, q


def rotmat_to_quat(R):
    """ 旋转矩阵 -> 四元数 (w,x,y,z) """
    q = np.empty((4,), dtype=np.float32)
    tr = np.trace(R)

    if tr > 0:
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
    return q / (np.linalg.norm(q) + 1e-12)


# ====================
# Numpy <-> Torch
# ====================
def to_numpy(t):
    return t.detach().cpu().numpy() if torch.is_tensor(t) else np.array(t)
