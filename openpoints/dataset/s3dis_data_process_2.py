import os
import numpy as np
import laspy
from tqdm import tqdm
import re
from sklearn.neighbors import NearestNeighbors  # 用于邻域搜索计算法向量

# 标签映射
label_mapping = {
    "铁塔": 0,
    "绝缘子": 0,
    "建筑物点": 1,
    "公路": 1,
    "低点": 1,
    "地面点": 1,
    "变电站": 1,
    "中等植被点": 1,
    "导线": 2,
    "引流线": 2,
    "地线": 2
}
class_names = {0: "前景(铁塔/绝缘子)", 1: "背景", 2: "前景(导线/地线)"}

# 法向量计算参数
NEIGHBOR_COUNT = 50  # 每个点用于计算法向量的邻域点数
NORMAL_CLIP_THRESHOLD = 1.0  # 法向量分量截断阈值

BETA = 1.25  # 背景 = BETA × (N0+N2)
BG_MIN = 3_000_000  # 背景保底总点数
MIN_KEEP_POINTS = 2000  # 背景文件保护阈值（≤这个点数不采样）


def compute_normals(coords, k=NEIGHBOR_COUNT):
    """
    通过邻域点的PCA分析计算法向量
    coords: 点云坐标数组，形状为 (N, 3)
    k: 邻域点数
    return: 法向量数组，形状为 (N, 3)，若点数不足则返回0向量
    """
    n_points = len(coords)
    if n_points < 3:  # 至少需要3个点才能计算法向量
        return np.zeros((n_points, 3), dtype=np.float32)

    # 调整邻域点数（如果总点数少于设定的邻域数）
    k = min(k, n_points - 1)

    # 构建KDTree查找邻域
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(coords)
    _, indices = nbrs.kneighbors(coords)  # 每个点的k个邻域索引

    normals = np.zeros((n_points, 3), dtype=np.float32)

    for i in tqdm(range(n_points), desc="计算法向量", leave=False):
        # 获取邻域点坐标
        neighbor_coords = coords[indices[i]]
        # 计算邻域中心点
        centroid = np.mean(neighbor_coords, axis=0)
        # 去中心化
        neighbor_centered = neighbor_coords - centroid
        # 计算协方差矩阵
        cov = np.cov(neighbor_centered.T)
        # 特征值分解（最小特征值对应的特征向量即为法向量）
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # 取最小特征值对应的特征向量作为法向量
        normal = eigenvectors[:, 0]
        # 归一化法向量
        normal_norm = np.linalg.norm(normal)
        if normal_norm > 1e-6:  # 避免除以零
            normal = normal / normal_norm
        # 截断异常值
        normal = np.clip(normal, -NORMAL_CLIP_THRESHOLD, NORMAL_CLIP_THRESHOLD)
        normals[i] = normal

    return normals


def merge_scene_files(s3dis_output_dir, train_ratio=0.7, val_ratio=0.1, seed=42):
    """把同一场景的多个类别文件合并成一个场景文件，再划分 train/val/test 场景"""
    raw_dir = os.path.join(s3dis_output_dir, "raw")
    merged_dir = os.path.join(s3dis_output_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)

    # 按场景分组（从raw文件夹读取）
    scene_groups = {}
    for fname in os.listdir(raw_dir):
        if fname.endswith(".npy") and fname.startswith("Area_"):
            m = re.match(r"(Area_\d+)", fname)
            if not m:
                raise ValueError(f"文件名不符合规则: {fname}")
            scene_id = m.group(1)  # e.g. "Area_14"
            scene_groups.setdefault(scene_id, []).append(os.path.join(raw_dir, fname))

    scene_files = []
    for scene_id, files in scene_groups.items():
        arrays = []
        for f in files:
            arr = np.load(f)
            label = int(arr[:, -1][0]) if arr.shape[1] >= 10 else -1  # 特征维度为10
            arr = ensure_feature_integrity(arr, label)
            arrays.append(arr)
        merged = np.concatenate(arrays, axis=0)
        save_path = os.path.join(merged_dir, f"{scene_id}.npy")
        np.save(save_path, merged)
        print(f"[合并场景] {scene_id}: {len(files)} 个文件 → {merged.shape[0]} 点")
        rel_path = os.path.relpath(save_path, s3dis_output_dir).replace("\\", "/")
        scene_files.append(rel_path)

    # 随机划分 train / val / test
    np.random.seed(seed)
    np.random.shuffle(scene_files)
    n_total = len(scene_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train_scenes = scene_files[:n_train]
    val_scenes = scene_files[n_train:n_train + n_val]
    test_scenes = scene_files[n_train + n_val:]

    def write_list(fname, data):
        with open(os.path.join(s3dis_output_dir, fname), "w") as f:
            f.write("\n".join(data))

    write_list("train_scenes.txt", train_scenes)
    write_list("val_scenes.txt", val_scenes)
    write_list("test_scenes.txt", test_scenes)

    print(f"[划分完成] 训练场景 {len(train_scenes)} 个, 验证场景 {len(val_scenes)} 个, 测试场景 {len(test_scenes)} 个")


def ensure_feature_integrity(arr, label):
    """确保每个点有10个特征（3坐标+3颜色+3法向量+1标签）"""
    if arr.shape[1] < 10:
        coords = arr[:, :3] if arr.shape[1] >= 3 else np.zeros((len(arr), 3), dtype=np.float32)
        colors = arr[:, 3:6] if arr.shape[1] >= 6 else np.ones((len(arr), 3), dtype=np.uint8) * 128
        normals = arr[:, 6:9] if arr.shape[1] >= 9 else np.zeros((len(arr), 3), dtype=np.float32)
        labels = np.full((len(arr), 1), label, dtype=np.int32)
        arr = np.hstack([coords, colors, normals, labels])
    elif arr.shape[1] > 10:
        arr = arr[:, :10]
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def count_points_per_file(las_input_dir):
    """统计每个文件各类别点数"""
    file_class_counts = {}
    total_class_counts = {0: 0, 1: 0, 2: 0}
    for fname in tqdm(os.listdir(las_input_dir), desc="统计点数"):
        if not fname.endswith(".las"):
            continue
        raw_label_name = fname.split("_")[-1].replace(".las", "")
        s3dis_label = label_mapping.get(raw_label_name, -1)
        if s3dis_label == -1:
            continue
        path = os.path.join(las_input_dir, fname)
        with laspy.open(path) as f:
            las = f.read()
            n = len(las.x)
        file_class_counts[fname] = {0: 0, 1: 0, 2: 0}
        file_class_counts[fname][s3dis_label] = n
        total_class_counts[s3dis_label] += n
    return file_class_counts, total_class_counts


def process_las_to_s3dis(las_input_dir, s3dis_output_dir, balance=True):
    raw_dir = os.path.join(s3dis_output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    file_class_counts, total_class_counts = count_points_per_file(las_input_dir)

    if balance:
        N0, N1, N2 = total_class_counts[0], total_class_counts[1], total_class_counts[2]
        target_background_total = min(
            N1,
            max(BG_MIN, int(BETA * (N0 + N2)))
        )
        print(f"\n类别总点数: {total_class_counts}")
        print(f"目标背景总点数: {target_background_total:,}")
        ratio = (N0 + N2) / target_background_total if target_background_total > 0 else 0
        print(f"(0+2) : 背景 ≈ {ratio:.3f}")

        total_bg_points = total_class_counts[1]
        bg_file_quota = {}
        for fname, counts in file_class_counts.items():
            if counts[1] > 0:
                ratio_file = counts[1] / total_bg_points
                bg_file_quota[fname] = int(target_background_total * ratio_file)
    else:
        bg_file_quota = None

    final_counts = {0: 0, 1: 0, 2: 0}

    for fname in tqdm(os.listdir(las_input_dir), desc="处理文件"):
        if not fname.endswith(".las"):
            continue

        raw_label_name = fname.split("_")[-1].replace(".las", "")
        s3dis_label = label_mapping.get(raw_label_name, -1)
        if s3dis_label == -1:
            print(f"[跳过] 无法识别标签: {fname}")
            continue

        path = os.path.join(las_input_dir, fname)
        with laspy.open(path) as f:
            las = f.read()
            # 提取坐标
            coords = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)

            # 处理颜色（先不计算法向量）
            color_list = []
            missing_channels = []
            if hasattr(las, "red"):
                color_list.append(las.red / 65535 * 255)
            else:
                color_list.append(np.ones(len(coords)) * 128)
                missing_channels.append("red")
            if hasattr(las, "green"):
                color_list.append(las.green / 65535 * 255)
            else:
                color_list.append(np.ones(len(coords)) * 128)
                missing_channels.append("green")
            if hasattr(las, "blue"):
                color_list.append(las.blue / 65535 * 255)
            else:
                color_list.append(np.ones(len(coords)) * 128)
                missing_channels.append("blue")
            colors = np.vstack(color_list).T.astype(np.uint8)

            if missing_channels:
                print(
                    f"[RGB Warning] {os.path.basename(path)} 缺失颜色通道: {', '.join(missing_channels)}，已用灰色填充")

        # 组合临时特征（法向量位置先用0填充）
        s3dis_data = np.hstack([
            coords,
            colors,
            np.zeros((len(coords), 3), dtype=np.float32),  # 临时填充0
            np.full((len(coords), 1), s3dis_label, dtype=np.int32)
        ])

        orig_points = len(s3dis_data)

        # 第一次特征检查
        s3dis_data = ensure_feature_integrity(s3dis_data, s3dis_label)

        # 背景类采样（裁剪）
        if balance and s3dis_label == 1:
            quota = bg_file_quota.get(fname, len(s3dis_data))
            if len(s3dis_data) > quota and len(s3dis_data) > MIN_KEEP_POINTS:
                idx = np.random.choice(len(s3dis_data), quota, replace=False)
                s3dis_data = s3dis_data[idx]
                print(
                    f"[采样] {fname} | 类别 {s3dis_label}({class_names[s3dis_label]}) | 原始: {orig_points} → 采样后: {len(s3dis_data)}")
            elif len(s3dis_data) <= MIN_KEEP_POINTS:
                print(
                    f"[跳过采样-小文件保护] {fname} | 类别 {s3dis_label}({class_names[s3dis_label]}) | 原始点数: {orig_points} (≤ {MIN_KEEP_POINTS})")
            else:
                print(f"[保留原样] {fname} | 类别 {s3dis_label}({class_names[s3dis_label]}) | 原始点数: {orig_points}")
        else:
            print(f"[保留原样] {fname} | 类别 {s3dis_label}({class_names[s3dis_label]}) | 原始点数: {orig_points}")

        # 裁剪后对保留的点计算法向量（核心修改）
        print(f"\n[法向量计算] 开始处理裁剪后的 {fname} (剩余 {len(s3dis_data)} 点)")
        coords_cropped = s3dis_data[:, :3]  # 从裁剪后的数据中提取坐标
        normals = compute_normals(coords_cropped)  # 仅对保留的点计算法向量
        s3dis_data[:, 6:9] = normals  # 将计算好的法向量替换临时填充的0

        # 最终特征检查
        s3dis_data = ensure_feature_integrity(s3dis_data, s3dis_label)

        # 法向量调试信息
        print(f"[调试] 法向量形状: {normals.shape} (点数×3)")
        if len(normals) > 0:
            print(f"[调试] 前3个点法向量示例:\n{normals[:3]}")
            print(f"[调试] 法向量范围: nx[{normals[:, 0].min():.4f}, {normals[:, 0].max():.4f}], "
                  f"ny[{normals[:, 1].min():.4f}, {normals[:, 1].max():.4f}], "
                  f"nz[{normals[:, 2].min():.4f}, {normals[:, 2].max():.4f}]")

        # 组合特征调试信息
        print(f"[调试] 组合后数据形状: {s3dis_data.shape} (应满足 点数×10)")
        if len(s3dis_data) > 0:
            print(f"[调试] 前2个点完整特征示例(坐标+颜色+法向量+标签):\n{s3dis_data[:2]}")

        final_counts[s3dis_label] += len(s3dis_data)

        # 保存到raw文件夹
        if not fname.startswith("Area_"):
            npy_name = f"Area_{fname.replace('.las', '.npy')}"
        else:
            npy_name = fname.replace(".las", ".npy")
        save_path = os.path.join(raw_dir, npy_name)
        np.save(save_path, s3dis_data)
        print(f"[保存] {save_path} ({len(s3dis_data)} 点，10维特征)\n")

    print("\n==== 最终采样后的类别点数分布 ====")
    total_points = sum(final_counts.values())
    for cls_id, count in final_counts.items():
        print(f"类别 {cls_id}({class_names[cls_id]}): {count:,} 点 ({count / total_points:.2%})")
    print(f"总点数: {total_points:,}")


if __name__ == "__main__":
    LAS_INPUT_DIR = r"D:\user\Documents\ai\paper\1_process\dataSet\s3dis_pointNeXt\input"
    S3DIS_OUTPUT_DIR = r"D:\user\Documents\ai\paper\1_process\dataSet\s3dis_pointNeXt\output"
    process_las_to_s3dis(LAS_INPUT_DIR, S3DIS_OUTPUT_DIR, balance=True)

    # 合并并划分
    merge_scene_files(S3DIS_OUTPUT_DIR, train_ratio=0.7)
