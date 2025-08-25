import os
import numpy as np
from tqdm import tqdm

# 标签名称映射（与你的预处理代码保持一致）
class_names = {
    0: "前景(铁塔/绝缘子)",
    1: "背景",
    2: "前景(导线/地线)"
}


def check_scene_labels(scene_dir):
    """
    检查场景目录下所有.npy文件的标签分布

    参数:
        scene_dir: 场景文件(.npy)所在的目录（通常是merged目录）
    """
    # 获取所有.npy文件
    scene_files = [f for f in os.listdir(scene_dir) if f.endswith(".npy")]

    if not scene_files:
        print(f"错误：在目录 {scene_dir} 中未找到任何.npy文件")
        return

    # 存储所有场景的标签分布
    all_scene_distributions = {}
    total_counts = {0: 0, 1: 0, 2: 0}

    print(f"开始检查 {len(scene_files)} 个场景文件...\n")

    # 遍历每个场景文件
    for fname in tqdm(scene_files, desc="检查进度"):
        file_path = os.path.join(scene_dir, fname)

        try:
            # 加载场景数据
            data = np.load(file_path)

            # 检查数据格式是否正确（是否包含10维特征）
            if data.ndim != 2 or data.shape[1] < 10:
                print(f"警告：{fname} 格式不正确，跳过（应为(N, 10)的数组）")
                continue

            # 提取标签列（第10列，索引9）
            labels = data[:, 9].astype(np.uint8)

            # 统计各类别数量
            count_0 = int((labels == 0).sum())
            count_1 = int((labels == 1).sum())
            count_2 = int((labels == 2).sum())
            total = count_0 + count_1 + count_2

            # 存储结果
            all_scene_distributions[fname] = {
                0: count_0,
                1: count_1,
                2: count_2,
                "total": total
            }

            # 累加至总计数
            total_counts[0] += count_0
            total_counts[1] += count_1
            total_counts[2] += count_2

        except Exception as e:
            print(f"处理 {fname} 时出错: {str(e)}")
            continue

    # 生成报告
    print("\n" + "=" * 50)
    print("场景标签分布检查报告")
    print("=" * 50)

    # 1. 每个场景的标签分布
    print("\n=== 每个场景标签分布 ===")
    for fname, dist in all_scene_distributions.items():
        print(f"{fname}: "
              f"0类({class_names[0]})={dist[0]}, "
              f"1类({class_names[1]})={dist[1]}, "
              f"2类({class_names[2]})={dist[2]}, "
              f"总点数={dist['total']}")

    # 2. 全量标签分布
    print("\n" + "=" * 50)
    print("=== 全部场景标签总分布 ===")
    total_all = total_counts[0] + total_counts[1] + total_counts[2]
    for cls_id in [0, 1, 2]:
        ratio = total_counts[cls_id] / total_all if total_all > 0 else 0
        print(f"类别 {cls_id}({class_names[cls_id]}): "
              f"{total_counts[cls_id]:,} 点 "
              f"({ratio:.2%})")
    print(f"总点数: {total_all:,}")

    # 3. 检查异常情况
    print("\n" + "=" * 50)
    print("=== 异常检查 ===")
    if total_counts[2] == 0:
        print("警告：所有场景中未发现2类标签（导线/地线/引流线）！")
    if total_counts[1] == 0:
        print("警告：所有场景中未发现1类标签（背景）！")

    # 检查是否有场景缺失某些标签
    for fname, dist in all_scene_distributions.items():
        missing = []
        if dist[0] == 0:
            missing.append(f"0类({class_names[0]})")
        if dist[1] == 0:
            missing.append(f"1类({class_names[1]})")
        if dist[2] == 0:
            missing.append(f"2类({class_names[2]})")
        if missing:
            print(f"场景 {fname} 缺失标签: {', '.join(missing)}")


if __name__ == "__main__":
    # 请修改为你的场景文件目录（通常是merged目录）
    SCENE_DIRECTORY = r"D:\user\Documents\ai\paper\1_process\dataSet\s3dis_pointNeXt\output\merged"
    # SCENE_DIRECTORY = r"D:\user\Documents\ai\paper\1_process\dataSet\s3dis_pointNeXt\s3dis_有法向量\merged"
    # 执行检查
    check_scene_labels(SCENE_DIRECTORY)
