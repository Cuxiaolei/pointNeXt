import os

# 预处理后的目录（包含所有 .npy 文件）
data_dir = r"D:\user\Documents\ai\paper\1_process\dataSet\s3dis_pointNeXt\output\raw"

# 遍历文件并修改名称
for filename in os.listdir(data_dir):
    if filename.endswith(".npy"):
        # 提取场景标识（如 1-2(1)）
        scene_id = filename.split("_")[0]  # 假设格式是 "1-2(1)_..."
        # 构造新文件名（添加 Area_1 前缀）
        new_filename = f"Area_1_{filename}"
        # 重命名
        os.rename(
            os.path.join(data_dir, filename),
            os.path.join(data_dir, new_filename)
        )