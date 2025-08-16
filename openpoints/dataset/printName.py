import os


def list_all_files(folder_path):
    """
    列出文件夹中所有文件的名称（包括子文件夹中的文件）

    参数:
        folder_path: 文件夹路径
    """
    # 检查文件夹路径是否存在
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在")
        return

    # 检查是否为文件夹
    if not os.path.isdir(folder_path):
        print(f"错误：'{folder_path}' 不是一个文件夹")
        return

    # 遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        # 输出当前目录
        print(f"\n目录: {root}")
        print("-" * 50)

        # 输出当前目录下的文件
        if files:
            for file in files:
                print(file)
        else:
            print("该目录下没有文件")


if __name__ == "__main__":
    # 在这里替换为你要查询的文件夹路径
    target_folder = r"D:\user\Documents\ai\paper\1_process\dataSet\s3dis_pointNeXt\output\merged"

    print(f"开始列出文件夹 '{target_folder}' 中的所有文件...\n")
    list_all_files(target_folder)
    print("\n文件列表输出完成")
