import os
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import laspy
from tqdm import tqdm

# --- Open3D 可选依赖：用于法线估计 ---
try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False

np.set_printoptions(suppress=True, precision=8)

############################################################
# 需求：只移除“全局中心/场景中心/单位球归一化”，保留 Open3D 法线估计
# 输出与原 MT 风格一致：coord.npy / color.npy / normal.npy / segment20.npy
############################################################

# 标签映射：保持 0/1/2 三类
CATEGORY_TO_SEGMENT: Dict[str, int] = {
    # 0 类（前景）
    "铁塔": 0,
    "绝缘子": 0,
    # 1 类（背景/附属）
    "建筑物点": 1,
    "公路": 1,
    "低点": 1,
    "地面点": 1,
    "变电站": 1,
    "中等植被点": 1,
    "交叉跨越下": 1,
    "不通航河流": 1,
    # 2 类（导线相关）
    "导线": 2,
    "地线": 2,
    "引流线": 2,
}

RANDOM_SEED = 42
DS_RATIO_1 = 0.8  # 类 1 的目标上限：<= 0.8 × (n0 + n1)
NORMAL_K = 15     # 法线估计 KNN 邻居数

############################################################
# 工具函数
############################################################

def _label_from_filename(fname: str) -> str:
    name = Path(fname).stem
    return name.split("_")[-1]


def _scene_id_from_filename(fname: str) -> str:
    # 以去掉末尾“_类别”的前缀作为场景 id，例如 a_b_c_铁塔.las → a_b_c
    name = Path(fname).stem
    parts = name.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:-1])
    return name


def group_files_by_scene(input_dir: Path) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for p in sorted(input_dir.glob("*.las")):
        sid = _scene_id_from_filename(p.name)
        groups.setdefault(sid, []).append(p)
    return groups


def load_xyz_rgb_label(las_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    las = laspy.read(str(las_path))
    xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)  # 原始坐标，不做任何中心化/归一化

    # 颜色：若为 16 位缩放到 0~255；缺失则用 128 填充
    if all(hasattr(las, ch) for ch in ("red", "green", "blue")):
        rgb = np.vstack((las.red, las.green, las.blue)).T.astype(np.float64)
        if rgb.max() > 255:
            rgb = (rgb / 65535.0 * 255.0)
        rgb = rgb.clip(0, 255).astype(np.uint8)
    else:
        rgb = np.full((xyz.shape[0], 3), 128, dtype=np.uint8)

    raw = _label_from_filename(las_path.name)
    if raw not in CATEGORY_TO_SEGMENT:
        raise ValueError(f"未知标签 '{raw}' 于文件: {las_path}")
    seg = np.full((xyz.shape[0],), CATEGORY_TO_SEGMENT[raw], dtype=np.uint8)
    return xyz, rgb, seg


def estimate_normals(coords: np.ndarray, k: int = NORMAL_K) -> np.ndarray:
    if coords.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if not _HAS_O3D:
        print("[WARN] 未检测到 Open3D，normal.npy 将写入全零。")
        return np.zeros((coords.shape[0], 3), dtype=np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    normals = np.asarray(pcd.normals, dtype=np.float64)
    normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
    return normals.astype(np.float32)


def downsample_class1_keep_0_2(labels: np.ndarray, ratio: float = DS_RATIO_1) -> np.ndarray:
    """保留全部 0 类与 2 类；将 1 类下采样到不超过 ratio × (n0 + n1)。"""
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]
    idx2 = np.where(labels == 2)[0]

    n0 = len(idx0)
    n1 = len(idx1)
    n2 = len(idx2)
    target = int(math.floor(ratio * (n0 + n2)))

    if n1 > target:
        rng = np.random.default_rng(RANDOM_SEED)
        idx1 = rng.choice(idx1, size=target, replace=False)

    keep = np.concatenate([idx0, idx1, idx2])
    keep.sort()
    return keep

############################################################
# 主流程
############################################################

def process_las_files_merge_scene(input_dir: str, output_dir: str, write_split: bool = True, train_ratio: float = 0.8) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    scene_groups = group_files_by_scene(input_path)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    scene_names: List[str] = []

    for idx, (scene_id, files) in enumerate(tqdm(scene_groups.items(), desc="Processing scenes")):
        scene_name = f"scene{idx:04d}_00"
        scene_dir = output_path / scene_name
        scene_dir.mkdir(exist_ok=True)

        xyz_list: List[np.ndarray] = []
        rgb_list: List[np.ndarray] = []
        seg_list: List[np.ndarray] = []

        for las_path in files:
            try:
                xyz, rgb, seg = load_xyz_rgb_label(las_path)
            except ValueError as e:
                print(f"[跳过] {e}")
                continue
            xyz_list.append(xyz)
            rgb_list.append(rgb)
            seg_list.append(seg)

        if not xyz_list:
            print(f"[跳过场景] {scene_id}（无有效点）")
            continue

        xyz = np.vstack(xyz_list)
        rgb = np.vstack(rgb_list)
        seg = np.concatenate(seg_list)

        # 下采样（逐场景）：保留 0/2，全量；1 类限制到 0.8×(n0+n1)
        keep_idx = downsample_class1_keep_0_2(seg, ratio=DS_RATIO_1)
        xyz = xyz[keep_idx]
        rgb = rgb[keep_idx]
        seg = seg[keep_idx]

        # 估计法线（基于原始坐标，不做任何中心化/归一化）
        normals = estimate_normals(xyz, k=NORMAL_K)

        # 保存四个文件
        np.save(scene_dir / "coord.npy", xyz.astype(np.float32))
        np.save(scene_dir / "color.npy", rgb.astype(np.uint8))
        np.save(scene_dir / "normal.npy", normals.astype(np.float32))
        np.save(scene_dir / "segment20.npy", seg.astype(np.uint8))

        scene_names.append(scene_name)

        # 简要统计
        n0 = int((seg == 0).sum())
        n1 = int((seg == 1).sum())
        n2 = int((seg == 2).sum())
        print(f"[{scene_name}] total={len(seg)}, 0={n0}, 1={n1}, 2={n2}")

    # 写 train/val 划分
    if write_split:
        random.shuffle(scene_names)
        n_total = len(scene_names)
        n_train = int(round(train_ratio * n_total))
        train_list = scene_names[:n_train]
        val_list = scene_names[n_train:]
        (output_path / "train.txt").write_text("\n".join(train_list), encoding="utf-8")
        (output_path / "val.txt").write_text("\n".join(val_list), encoding="utf-8")
        print(f"\n[完成] 共 {n_total} 个场景 → train: {len(train_list)} / val: {len(val_list)}")


if __name__ == "__main__":
    # === 按需修改你的输入/输出路径 ===
    INPUT_DIR = r"D:\三维重建\点云资料\输电人工智能\las"
    OUTPUT_DIR = r"D:\三维重建\数据集\scenenet\Scencnet03"
    process_las_files_merge_scene(INPUT_DIR, OUTPUT_DIR, write_split=True, train_ratio=0.8)
