import os


def print_directory_structure(root_dir, prefix=""):
    """
    以树形结构打印指定目录的所有文件和子目录

    参数:
        root_dir: 要遍历的根目录路径
        prefix: 用于构建树形结构的前缀字符串
    """
    # 检查目录是否存在
    if not os.path.exists(root_dir):
        print(f"错误: 目录 '{root_dir}' 不存在")
        return

    if not os.path.isdir(root_dir):
        print(f"错误: '{root_dir}' 不是一个目录")
        return

    # 获取目录中的所有项目（文件和子目录）
    items = os.listdir(root_dir)

    # 遍历所有项目
    for index, item in enumerate(items):
        item_path = os.path.join(root_dir, item)
        # 判断是否为最后一个项目，用于调整连接线
        is_last = index == len(items) - 1

        # 打印当前项目
        if is_last:
            print(f"{prefix}└── {item}")
            # 为下一级准备前缀
            new_prefix = f"{prefix}    "
        else:
            print(f"{prefix}├── {item}")
            # 为下一级准备前缀
            new_prefix = f"{prefix}│   "

        # 如果是目录，递归处理
        if os.path.isdir(item_path):
            print_directory_structure(item_path, new_prefix)


if __name__ == "__main__":
    # 在这里直接指定要遍历的目录路径
    # 例如:
    # Windows系统: target_dir = "C:\\Users\\YourName\\Documents"
    # macOS/Linux系统: target_dir = "/home/yourname/documents"
    target_dir = r"D:\user\Documents\ai\paper\1_process\dataSet\s3dis_pointNeXt\output"

    print(f"目录结构: {target_dir}")
    print_directory_structure(target_dir)
'''
├── merged
│   ├── Area_1.npy
│   ├── Area_10.npy
│   ├── Area_11.npy
│   ├── Area_12.npy
│   ├── Area_13.npy
│   ├── Area_14.npy
│   ├── Area_15.npy
│   ├── Area_16.npy
│   ├── Area_17.npy
│   ├── Area_18.npy
│   ├── Area_19.npy
│   ├── Area_2.npy
│   ├── Area_20.npy
│   ├── Area_21.npy
│   ├── Area_22.npy
│   ├── Area_23.npy
│   ├── Area_24.npy
│   ├── Area_25.npy
│   ├── Area_26.npy
│   ├── Area_27.npy
│   ├── Area_28.npy
│   ├── Area_29.npy
│   ├── Area_3.npy
│   ├── Area_30.npy
│   ├── Area_4.npy
│   ├── Area_5.npy
│   ├── Area_6.npy
│   ├── Area_7.npy
│   ├── Area_8.npy
│   └── Area_9.npy
├── test_scenes.txt
├── train_scenes.txt
└── val_scenes.txt


'''