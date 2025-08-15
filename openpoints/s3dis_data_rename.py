import os

# 文件所在的目录路径，根据实际情况修改
folder_path = r"/root/autodl-tmp/raw"

# 遍历目录下的文件
for filename in os.listdir(folder_path):
    if filename.endswith(".npy"):
        # 分割文件名，处理以 Area_ 开头的情况
        if filename.startswith("Area_"):
            # 按 '_' 分割，取从第 2 个元素开始的部分重新拼接
            parts = filename.split("_")
            new_filename = "_".join(["Area"] + parts[2:])
            old_filepath = os.path.join(folder_path, filename)
            new_filepath = os.path.join(folder_path, new_filename)
            # 重命名文件
            os.rename(old_filepath, new_filepath)
            print(f"已重命名: {filename} -> {new_filename}")