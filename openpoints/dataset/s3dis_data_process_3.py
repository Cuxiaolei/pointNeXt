import os
import numpy as np
import laspy
import shutil
from tqdm import tqdm
import re
from sklearn.neighbors import NearestNeighbors

# --- Open3D 可选依赖 ---
try:
    import open3d as o3d

    _HAS_O3D = True
except Exception:
    _HAS_O3D = False

# 标签映射
label_mapping = {
    # 0 类（前景）
    "铁塔": 0, "绝缘子": 0,
    # 1 类（背景/附属）
    "建筑物点": 1, "公路": 1, "低点": 1, "地面点": 1,
    "变电站": 1, "中等植被点": 1, "交叉跨越下": 1, "不通航河流": 1,
    # 2 类（导线相关）
    "导线": 2, "引流线": 2, "地线": 2
}
class_names = {0: "前景(铁塔/绝缘子)", 1: "背景", 2: "前景(导线/地线)"}

# 法向量计算参数
NORMAL_K = 15
DS_RATIO_1 = 0.8
RANDOM_SEED = 42


def compute_normals(coords, k=NORMAL_K):
    if coords.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if not _HAS_O3D:
        print("[WARN] 未检测到 Open3D，法向量将填充为0。")
        return np.zeros((coords.shape[0], 3), dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    normals = np.asarray(pcd.normals, dtype=np.float64)
    normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
    return normals.astype(np.float32)


def downsample_class1_keep_0_2(labels: np.ndarray, ratio: float = DS_RATIO_1) -> np.ndarray:
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]
    idx2 = np.where(labels == 2)[0]

    n0 = len(idx0)
    n1 = len(idx1)
    n2 = len(idx2)
    target = int(np.floor(ratio * (n0 + n2)))

    if (n0 + n2) == 0:
        target = 0
        print("[注意] 场景中0类+2类点数为0，1类点将全部过滤")

    if n1 > target:
        rng = np.random.default_rng(RANDOM_SEED)
        idx1 = rng.choice(idx1, size=target, replace=False)

    keep = np.concatenate([idx0, idx1, idx2])
    keep.sort()
    return keep


def merge_scene_files(s3dis_output_dir, train_ratio=0.7, val_ratio=0.1, seed=42):
    raw_dir = os.path.join(s3dis_output_dir, "raw")
    merged_dir = os.path.join(s3dis_output_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)

    # 按场景分组（提取纯数字编号）
    scene_groups = {}
    for fname in os.listdir(raw_dir):
        if fname.endswith(".npy") and fname.startswith("Area_"):
            # 提取Area后的数字部分（支持"Area_1-2(1_2)_xxx.npy"等格式）
            num_match = re.search(r"Area_(\d+)", fname)
            if not num_match:
                print(f"[跳过] 文件名不符合规则（无法提取数字编号）: {fname}")
                continue

            # 场景编号（仅保留数字部分）
            area_num = num_match.group(1)
            scene_id = f"Area_{area_num}"  # 统一命名为Area_数字（如Area_1）
            file_path = os.path.join(raw_dir, fname)

            # 检查文件是否为空
            arr = np.load(file_path)
            if arr.size == 0:
                print(f"[跳过空文件] {fname}")
                continue

            scene_groups.setdefault(scene_id, []).append(file_path)

    scene_files = []
    global_final_counts = {0: 0, 1: 0, 2: 0}
    print("\n==== 开始合并场景 ====")

    for scene_id, files in scene_groups.items():
        if not files:
            print(f"[跳过空场景] {scene_id}")
            continue

        arrays = []
        scene_class_counts = {0: 0, 1: 0, 2: 0}
        for f in files:
            arr = np.load(f)
            if arr.size == 0:
                print(f"[跳过空文件] {os.path.basename(f)}")
                continue

            file_label = int(arr[:, -1][0]) if arr.shape[1] >= 10 and len(arr) > 0 else -1
            if file_label in scene_class_counts:
                scene_class_counts[file_label] += len(arr)
                print(
                    f"  包含文件: {os.path.basename(f)} | 标签: {file_label}({class_names[file_label]}) | 点数: {len(arr):,}")
            else:
                print(f"  [警告] 文件{os.path.basename(f)}包含未知标签{file_label}，跳过")
                continue

            arr = ensure_feature_integrity(arr, file_label)
            arrays.append(arr)

        if not arrays:
            print(f"[跳过空场景] {scene_id}")
            continue

        # 合并场景并下采样
        merged_raw = np.concatenate(arrays, axis=0)
        print(f"\n[合并场景] {scene_id}: {len(files)} 个文件 → 原始总点数: {merged_raw.shape[0]:,}")
        print(
            f"  原始标签分布: 0类{scene_class_counts[0]:,}, 1类{scene_class_counts[1]:,}, 2类{scene_class_counts[2]:,}")

        # 下采样
        scene_labels = merged_raw[:, 9].astype(np.uint8)
        keep_idx = downsample_class1_keep_0_2(scene_labels)
        merged_downsampled = merged_raw[keep_idx]

        # 重新计算法向量
        coords_downsampled = merged_downsampled[:, :3]
        normals_downsampled = compute_normals(coords_downsampled)
        merged_downsampled[:, 6:9] = normals_downsampled

        # 统计下采样后分布
        down_0 = int((merged_downsampled[:, 9] == 0).sum())
        down_1 = int((merged_downsampled[:, 9] == 1).sum())
        down_2 = int((merged_downsampled[:, 9] == 2).sum())
        global_final_counts[0] += down_0
        global_final_counts[1] += down_1
        global_final_counts[2] += down_2
        print(f"  下采样后分布: 0类{down_0:,}, 1类{down_1:,}, 2类{down_2:,} | 总点数{merged_downsampled.shape[0]:,}")

        # 保存为指定格式（Area_数字.npy）
        save_path = os.path.join(merged_dir, f"{scene_id}.npy")
        np.save(save_path, merged_downsampled)
        rel_path = os.path.relpath(save_path, s3dis_output_dir).replace("\\", "/")
        scene_files.append(rel_path)
        print(f"[保存场景] {save_path} 已保存")

    # 打印全量分布
    print("\n==== 全量场景下采样后的类别点数分布 ====")
    total_points = sum(global_final_counts.values())
    for cls_id in [0, 1, 2]:
        ratio = global_final_counts[cls_id] / total_points if total_points > 0 else 0
        print(f"类别 {cls_id}({class_names[cls_id]}): {global_final_counts[cls_id]:,} 点 ({ratio:.2%})")
    print(f"全量总点数: {total_points:,}")

    # 划分数据集
    if not scene_files:
        print("[警告] 无有效场景，跳过划分")
        return

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

    print(
        f"\n[划分完成] 训练场景 {len(train_scenes)} 个, 验证场景 {len(val_scenes)} 个, 测试场景 {len(test_scenes)} 个")

    # 删除raw目录及其内容
    if os.path.exists(raw_dir):
        shutil.rmtree(raw_dir)
        print(f"\n[清理完成] 已删除临时目录: {raw_dir}")


def ensure_feature_integrity(arr, label):
    if arr.shape[1] < 10:
        coords = arr[:, :3] if arr.shape[1] >= 3 else np.zeros((len(arr), 3), dtype=np.float32)
        colors = arr[:, 3:6] if arr.shape[1] >= 6 else np.ones((len(arr), 3), dtype=np.uint8) * 128
        normals = arr[:, 6:9] if arr.shape[1] >= 9 else np.zeros((len(arr), 3), dtype=np.float32)
        labels = np.full((len(arr), 1), label, dtype=np.uint8)
        arr = np.hstack([coords, colors, normals, labels])
    elif arr.shape[1] > 10:
        arr = arr[:, :10]
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def count_points_per_file(las_input_dir):
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


def process_las_to_s3dis(las_input_dir, s3dis_output_dir):
    raw_dir = os.path.join(s3dis_output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    file_class_counts, total_class_counts = count_points_per_file(las_input_dir)

    print(f"\n==== 全量原始类别点数统计 ====")
    print(f"类别总点数: 0类: {total_class_counts[0]:,}, 1类: {total_class_counts[1]:,}, 2类: {total_class_counts[2]:,}")
    print(f"全量原始总点数: {sum(total_class_counts.values()):,}")

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

            # 颜色处理
            has_red = hasattr(las, "red")
            has_green = hasattr(las, "green")
            has_blue = hasattr(las, "blue")

            red = las.red if has_red else np.ones(len(coords), dtype=np.uint16) * (128 * 256)
            green = las.green if has_green else np.ones(len(coords), dtype=np.uint16) * (128 * 256)
            blue = las.blue if has_blue else np.ones(len(coords), dtype=np.uint16) * (128 * 256)

            rgb = np.vstack([red, green, blue]).T.astype(np.float64)
            if rgb.max() > 255:
                rgb = (rgb / 65535.0 * 255.0)
            rgb = rgb.clip(0, 255).astype(np.uint8)

            missing_channels = []
            if not has_red:
                missing_channels.append("red")
            if not has_green:
                missing_channels.append("green")
            if not has_blue:
                missing_channels.append("blue")
            if missing_channels:
                print(
                    f"[RGB Warning] {os.path.basename(path)} 缺失颜色通道: {', '.join(missing_channels)}，已用灰色填充")

        # 生成标签列
        labels = np.full((len(coords), 1), s3dis_label, dtype=np.uint8)

        # 组合特征
        s3dis_data = np.hstack([
            coords,
            rgb,
            np.zeros((len(coords), 3), dtype=np.float32),
            labels
        ])

        s3dis_data = ensure_feature_integrity(s3dis_data, s3dis_label)
        normals = compute_normals(coords)
        s3dis_data[:, 6:9] = normals

        print(f"\n[处理文件] {fname} | 类别 {s3dis_label}({class_names[s3dis_label]}) | 保存点数: {len(s3dis_data):,}")

        # 保存到raw文件夹（保留原始文件名用于后续分组）
        npy_name = fname.replace(".las", ".npy")
        save_path = os.path.join(raw_dir, npy_name)
        np.save(save_path, s3dis_data)
        print(f"[保存文件] {save_path} 已保存")


if __name__ == "__main__":
    LAS_INPUT_DIR = r"D:\user\Documents\ai\paper\1_process\dataSet\s3dis_pointNeXt\input"
    S3DIS_OUTPUT_DIR = r"D:\user\Documents\ai\paper\1_process\dataSet\s3dis_pointNeXt\output"

    # 1. 处理LAS文件为临时NPY（保存到raw目录）
    process_las_to_s3dis(LAS_INPUT_DIR, S3DIS_OUTPUT_DIR)

    # 2. 合并场景+下采样+划分数据集+删除raw目录
    merge_scene_files(S3DIS_OUTPUT_DIR, train_ratio=0.7)
