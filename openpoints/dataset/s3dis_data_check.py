import os
import numpy as np

raw_dir = r"D:\user\Documents\ai\paper\1_process\dataSet\s3dis_pointNeXt\output\raw"
for fname in os.listdir(raw_dir):
    if "Area_5" in fname and fname.endswith(".npy"):
        path = os.path.join(raw_dir, fname)
        data = np.load(path)
        if data.size == 0:  # 空数组
            print(f"空文件: {fname}")