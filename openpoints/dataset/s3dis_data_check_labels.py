import os
import numpy as np

# 你的数据根目录
data_root = r"D:\user\Documents\ai\paper\1_process\dataSet\s3dis_pointNeXt\output\merged"

def check_labels(data_root):
    npy_files = sorted([f for f in os.listdir(data_root) if f.endswith(".npy")])

    summary = {}
    for f in npy_files:
        path = os.path.join(data_root, f)
        data = np.load(path)
        labels = data[:, 6].astype(int)
        uniq, cnts = np.unique(labels, return_counts=True)
        summary[f] = dict(zip(uniq.tolist(), cnts.tolist()))

    print("=== 每个场景标签分布 ===")
    for f, dist in summary.items():
        print(f"{f}: {dist}")

    # 统计整体
    total = {}
    for dist in summary.values():
        for k, v in dist.items():
            total[k] = total.get(k, 0) + v
    print("\n=== 全部场景标签总分布 ===")
    print(total)


if __name__ == "__main__":
    check_labels(data_root)
