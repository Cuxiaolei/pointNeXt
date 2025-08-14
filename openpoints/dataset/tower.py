import os
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

__all__ = ['TowerDataset']

class TowerDataset(Dataset):
    def __init__(self,
                 split='train',
                 data_root='data/tower',
                 num_points=8000,
                 feat_keys=['xyz', 'normal', 'rgb'],
                 classes=3,
                 augment=False,
                 **kwargs):
        """
        PointNeXt dataset format for Tower point cloud segmentation.
        """
        self.split = split
        self.data_root = Path(data_root) / split
        self.num_points = num_points
        self.feat_keys = feat_keys
        self.classes = classes
        self.augment = augment

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")

        # 场景文件夹列表
        self.scenes = sorted([d for d in self.data_root.iterdir() if d.is_dir()])
        if len(self.scenes) == 0:
            raise RuntimeError(f"No scenes found in {self.data_root}")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_path = self.scenes[idx]

        # 加载必要数据
        coord = np.load(scene_path / "coord.npy").astype(np.float32)  # xyz
        normal = np.load(scene_path / "normal.npy").astype(np.float32) if (scene_path / "normal.npy").exists() else None
        rgb = np.load(scene_path / "color.npy").astype(np.float32) / 255.0 if (scene_path / "color.npy").exists() else None
        label = np.load(scene_path / "label.npy").astype(np.int64)

        # 采样/补齐
        N = coord.shape[0]
        if N >= self.num_points:
            choice = np.random.choice(N, self.num_points, replace=False)
        else:
            pad = np.random.choice(N, self.num_points - N, replace=True)
            choice = np.concatenate([np.arange(N), pad])

        coord = coord[choice]
        label = label[choice]
        if normal is not None: normal = normal[choice]
        if rgb is not None: rgb = rgb[choice]

        # 数据增强
        if self.augment:
            # 随机旋转（Z轴）
            theta = np.random.uniform(-np.pi/12, np.pi/12)
            c, s = np.cos(theta), np.sin(theta)
            Rz = np.array([[c, -s, 0],
                           [s,  c, 0],
                           [0,  0, 1]], dtype=np.float32)
            coord = coord @ Rz.T
            if normal is not None:
                normal = normal @ Rz.T

            # 随机缩放
            scale = np.random.uniform(0.9, 1.1)
            coord *= scale

            # 均值扰动
            coord += np.random.normal(0, 0.005, coord.shape)
            if rgb is not None:
                rgb = np.clip(rgb + np.random.normal(0, 0.02, rgb.shape), 0, 1)

        # 按 feat_keys 拼接特征
        feats = []
        for key in self.feat_keys:
            if key == 'xyz':
                feats.append(coord)
            elif key == 'normal' and normal is not None:
                feats.append(normal)
            elif key == 'rgb' and rgb is not None:
                feats.append(rgb)
        feat = np.concatenate(feats, axis=1).astype(np.float32)

        return coord, feat, label
