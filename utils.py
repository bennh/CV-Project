import numpy as np


# ---------------- 四元数 & 矩阵转换 ----------------

def quaternion_to_matrix(q):
    """
    四元数 → 旋转矩阵
    Args:
        q: [4] numpy array, (x, y, z, w)
    Returns:
        R: [3,3] rotation matrix
    """
    x, y, z, w = q
    R = np.array([
        [1 - 2 * (y ** 2 + z ** 2),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
    ], dtype=np.float32)
    return R


def matrix_to_quaternion(R):
    """
    旋转矩阵 → 四元数
    Args:
        R: [3,3] rotation matrix
    Returns:
        q: [4] (x, y, z, w)
    """
    trace = np.trace(R)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        # 找到对角元素最大的一行，避免数值不稳定
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
            qw = (R[2, 1] - R[1, 2]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
            qw = (R[0, 2] - R[2, 0]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
            qw = (R[1, 0] - R[0, 1]) / s

    q = np.array([qx, qy, qz, qw], dtype=np.float32)
    return q / np.linalg.norm(q)


def pose_to_tq(pose_mat):
    """
    4x4 位姿矩阵 → (translation, quaternion)
    """
    t = pose_mat[:3, 3]
    R = pose_mat[:3, :3]
    q = matrix_to_quaternion(R)
    return t, q


def tq_to_pose(t, q):
    """
    (translation, quaternion) → 4x4 位姿矩阵
    """
    R = quaternion_to_matrix(q)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose


# ---------------- 误差计算 ----------------

def translation_error(t_gt, t_pred):
    """
    平移误差 (欧几里得距离, 单位: 米)
    """
    return np.linalg.norm(t_gt - t_pred)


def rotation_error(q_gt, q_pred):
    """
    旋转误差 (单位: 度)
    """
    # 归一化
    q_gt = q_gt / np.linalg.norm(q_gt)
    q_pred = q_pred / np.linalg.norm(q_pred)

    # 四元数内积对应夹角
    dot = np.abs(np.dot(q_gt, q_pred))
    dot = np.clip(dot, -1.0, 1.0)
    theta = 2 * np.arccos(dot)  # 弧度
    return np.degrees(theta)


def pose_error(pose_gt, pose_pred):
    """
    综合计算平移 & 旋转误差
    Args:
        pose_gt: [4,4] GT pose
        pose_pred: [4,4] Pred pose
    Returns:
        t_err: 平移误差 (m)
        r_err: 旋转误差 (deg)
    """
    t_gt, q_gt = pose_to_tq(pose_gt)
    t_pred, q_pred = pose_to_tq(pose_pred)

    t_err = translation_error(t_gt, t_pred)
    r_err = rotation_error(q_gt, q_pred)

    return t_err, r_err


# ---------------- Demo ----------------
if __name__ == "__main__":
    # 随机 pose
    t = np.array([1.0, 2.0, 3.0])
    q = np.array([0.0, 0.0, 0.0, 1.0])  # 单位四元数
    pose = tq_to_pose(t, q)

    # 转换检查
    t2, q2 = pose_to_tq(pose)
    print("Original t:", t, "Recovered t:", t2)
    print("Original q:", q, "Recovered q:", q2)

    # 误差测试
    pose_pred = tq_to_pose(t + np.array([0.1, -0.1, 0.05]),
                           np.array([0.0, 0.1, 0.0, 0.99]))
    t_err, r_err = pose_error(pose, pose_pred)
    print(f"Translation Error = {t_err:.4f} m, Rotation Error = {r_err:.2f}°")
