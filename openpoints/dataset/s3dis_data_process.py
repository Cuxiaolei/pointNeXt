import os
import numpy as np
import laspy
from tqdm import tqdm
from collections import defaultdict

# ================== 配置 ==================
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
# 背景 = BETA × (N0 + N2)；等价于 (0+2) = 0.8 × 背景 ⇒ BETA = 1.25
BETA = 1.25
BG_MIN = 3_000_000    # 背景保底点数，防止太少
# =========================================

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
    s3dis_raw_dir = os.path.join(s3dis_output_dir, "raw")
    os.makedirs(s3dis_raw_dir, exist_ok=True)

    file_class_counts, total_class_counts = count_points_per_file(las_input_dir)

    if balance:
        N0, N1, N2 = total_class_counts[0], total_class_counts[1], total_class_counts[2]
        target_background_total = min(
            N1,  # 不超过原始背景总量
            max(BG_MIN, int(BETA * (N0 + N2)))  # 背景 = 1.25 × (0+2)，保底
        )
        print(f"\n类别总点数: {total_class_counts}")
        print(f"目标背景总点数: {target_background_total:,}")
        # 也打印一下 (0+2)/(目标背景) ，应接近 0.8
        ratio = (N0 + N2) / target_background_total if target_background_total > 0 else 0
        print(f"(0+2) : 背景 ≈ {ratio:.3f}")

        # 按文件比例分配背景配额（保证每个文件都有样本）
        total_bg_points = total_class_counts[1]
        bg_file_quota = {}
        for fname, counts in file_class_counts.items():
            if counts[1] > 0:
                ratio_file = counts[1] / total_bg_points
                bg_file_quota[fname] = int(target_background_total * ratio_file)
    else:
        bg_file_quota = None

    # 最终统计
    final_counts = {0: 0, 1: 0, 2: 0}

    for fname in tqdm(os.listdir(las_input_dir), desc="处理文件"):
        if not fname.endswith(".las"):
            continue

        raw_label_name = fname.split("_")[-1].replace(".las", "")
        s3dis_label = label_mapping.get(raw_label_name, -1)
        if s3dis_label == -1:
            print(f"警告：无法识别 {fname} 的标签，已跳过")
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

        # 背景类采样（按文件配额）
        if balance and s3dis_label == 1:
            quota = bg_file_quota.get(fname, len(s3dis_data))
            if len(s3dis_data) > quota:
                idx = np.random.choice(len(s3dis_data), quota, replace=False)
                s3dis_data = s3dis_data[idx]

        final_counts[s3dis_label] += len(s3dis_data)

        output_npy_path = os.path.join(s3dis_raw_dir, fname.replace(".las", ".npy"))
        np.save(output_npy_path, s3dis_data)

    # 输出最终比例
    print("\n==== 最终采样后的类别点数分布 ====")
    total_points = sum(final_counts.values())
    for cls_id, count in final_counts.items():
        pct = (count / total_points) if total_points > 0 else 0.0
        print(f"类别 {cls_id}: {count:,} 点 ({pct:.2%})")
    print(f"总点数: {total_points:,}")

if __name__ == "__main__":
    LAS_INPUT_DIR = r"D:\user\Documents\ai\paper\1_process\dataSet\s3dis_pointNeXt\input"
    S3DIS_OUTPUT_DIR = r"D:\user\Documents\ai\paper\1_process\dataSet\s3dis_pointNeXt\output"
    process_las_to_s3dis(LAS_INPUT_DIR, S3DIS_OUTPUT_DIR, balance=True)
