import os
import numpy as np
import laspy
from tqdm import tqdm

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

BETA = 1.25                # 背景 = BETA × (N0+N2)
BG_MIN = 3_000_000         # 背景保底总点数
MIN_KEEP_POINTS = 2000     # 背景文件保护阈值（≤这个点数不采样）

def ensure_feature_integrity(arr, label):
    """确保每个点有7个特征（3坐标+3颜色+1标签）"""
    if arr.shape[1] < 7:
        coords = arr[:, :3] if arr.shape[1] >= 3 else np.zeros((len(arr), 3), dtype=np.float32)
        colors = arr[:, 3:6] if arr.shape[1] >= 6 else np.ones((len(arr), 3), dtype=np.uint8) * 128
        labels = np.full((len(arr), 1), label, dtype=np.int32)
        arr = np.hstack([coords, colors, labels])
    elif arr.shape[1] > 7:
        arr = arr[:, :7]
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
            coords = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
            if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
                colors = np.vstack([
                    las.red / 65535 * 255,
                    las.green / 65535 * 255,
                    las.blue / 65535 * 255
                ]).T.astype(np.uint8)
            else:
                colors = np.ones((len(coords), 3), dtype=np.uint8) * 128

        s3dis_data = np.hstack([
            coords,
            colors,
            np.full((len(coords), 1), s3dis_label, dtype=np.int32)
        ])
        orig_points = len(s3dis_data)

        # 第一次特征检查
        s3dis_data = ensure_feature_integrity(s3dis_data, s3dis_label)

        # 背景类采样（保护小文件）
        if balance and s3dis_label == 1:
            quota = bg_file_quota.get(fname, len(s3dis_data))
            if len(s3dis_data) > quota and len(s3dis_data) > MIN_KEEP_POINTS:
                idx = np.random.choice(len(s3dis_data), quota, replace=False)
                s3dis_data = s3dis_data[idx]
                print(f"[采样] {fname} | 类别 {s3dis_label}({class_names[s3dis_label]}) | 原始: {orig_points} → 采样后: {len(s3dis_data)}")
            elif len(s3dis_data) <= MIN_KEEP_POINTS:
                print(f"[跳过采样-小文件保护] {fname} | 类别 {s3dis_label}({class_names[s3dis_label]}) | 原始点数: {orig_points} (≤ {MIN_KEEP_POINTS})")
            else:
                print(f"[保留原样] {fname} | 类别 {s3dis_label}({class_names[s3dis_label]}) | 原始点数: {orig_points}")
        else:
            print(f"[保留原样] {fname} | 类别 {s3dis_label}({class_names[s3dis_label]}) | 原始点数: {orig_points}")

        # 最终特征检查
        s3dis_data = ensure_feature_integrity(s3dis_data, s3dis_label)

        final_counts[s3dis_label] += len(s3dis_data)

        if not fname.startswith("Area_"):
            npy_name = f"Area_{fname.replace('.las', '.npy')}"
        else:
            npy_name = fname.replace(".las", ".npy")
        save_path = os.path.join(raw_dir, npy_name)
        np.save(save_path, s3dis_data)
        print(f"[保存] {save_path} ({len(s3dis_data)} 点)\n")

    print("\n==== 最终采样后的类别点数分布 ====")
    total_points = sum(final_counts.values())
    for cls_id, count in final_counts.items():
        print(f"类别 {cls_id}({class_names[cls_id]}): {count:,} 点 ({count / total_points:.2%})")
    print(f"总点数: {total_points:,}")

if __name__ == "__main__":
    LAS_INPUT_DIR = r"D:\user\Documents\ai\paper\1_process\dataSet\s3dis_pointNeXt\input"
    S3DIS_OUTPUT_DIR = r"D:\user\Documents\ai\paper\1_process\dataSet\s3dis_pointNeXt\output"
    process_las_to_s3dis(LAS_INPUT_DIR, S3DIS_OUTPUT_DIR, balance=True)
