import os
import numpy as np
import laspy
from tqdm import tqdm

# 你的标签映射规则（严格按需求）
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


# S3DIS 数据集的格式要求：每个文件保存为 [x, y, z, r, g, b, label]
def process_las_to_s3dis(las_input_dir, s3dis_output_dir):
    """
    将所有 .las 文件转换为 S3DIS 格式的 .npy 文件
    :param las_input_dir: 你的 .las 文件所在目录（如包含 1-2(1)_变电站.las）
    :param s3dis_output_dir: 输出的 S3DIS 格式数据目录（会生成 raw/ 子目录）
    """
    # 创建 S3DIS 标准的 raw 目录（与 s3dis.py 加载逻辑对齐）
    s3dis_raw_dir = os.path.join(s3dis_output_dir, "raw")
    os.makedirs(s3dis_raw_dir, exist_ok=True)

    # 遍历所有 .las 文件
    for las_filename in tqdm(os.listdir(las_input_dir)):
        if not las_filename.endswith(".las"):
            continue  # 跳过非 .las 文件

        # 解析文件名中的标签名（如 "1-2(1)_变电站.las" → 提取 "变电站"）
        raw_label_name = las_filename.split("_")[-1].replace(".las", "")
        # 映射到你的三分类标签（0/1/2）
        s3dis_label = label_mapping.get(raw_label_name, -1)
        if s3dis_label == -1:
            print(f"警告：无法识别 {las_filename} 的标签，已跳过")
            continue

        # 读取 .las 文件
        las_path = os.path.join(las_input_dir, las_filename)
        with laspy.open(las_path) as f:
            las = f.read()
            # 提取坐标 (x, y, z)
            coords = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
            # 提取颜色 (r, g, b) → 转换为 0-255 的 uint8
            if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
                colors = np.vstack([
                    las.red / 65535 * 255,  # LAS 颜色是 16 位，转 8 位
                    las.green / 65535 * 255,
                    las.blue / 65535 * 255
                ]).T.astype(np.uint8)
            else:
                # 无颜色时，用灰色填充
                colors = np.ones((len(coords), 3), dtype=np.uint8) * 128

        # 构造 S3DIS 格式的数据：[x, y, z, r, g, b, label]
        # 注意：S3DIS 的 label 是 int32 类型
        s3dis_data = np.hstack([
            coords,
            colors,
            np.full((len(coords), 1), s3dis_label, dtype=np.int32)
        ])

        # 保存为 .npy 文件（与 s3dis.py 加载逻辑对齐）
        # 文件名格式：保持与原始场景关联（如 "1-2(1)_变电站.npy"）
        output_npy_path = os.path.join(
            s3dis_raw_dir,
            las_filename.replace(".las", ".npy")
        )
        np.save(output_npy_path, s3dis_data)
        print(f"已处理 {las_filename} → 保存至 {output_npy_path}")


if __name__ == "__main__":
    # ========== 请修改为你的实际路径 ==========
    LAS_INPUT_DIR = r"D:\user\Documents\ai\paper\1_process\dataSet\s3dis_pointNeXt\input"  # 你的 .las 文件所在文件夹
    S3DIS_OUTPUT_DIR = r"D:\user\Documents\ai\paper\1_process\dataSet\s3dis_pointNeXt\output"  # 输出的 S3DIS 格式数据目录
    # =========================================

    process_las_to_s3dis(LAS_INPUT_DIR, S3DIS_OUTPUT_DIR)