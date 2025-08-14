import os
import numpy as np
from torch.utils.data import Dataset
from .dataset_base import DatasetBase

__all__ = ['TowerDataset']

class TowerDataset(DatasetBase):
    def __init__(self, split='train', data_root='data/tower', num_points=8000, feat_keys=['xyz', 'normal', 'rgb'], **kwargs):
        super().__init__()
        self.split = split
        self.data_root = os.path.join(data_root, split)
        self.num_points = num_points
        self.feat_keys = feat_keys

        self.scenes = sorted(os.listdir(self.data_root))
        self.coord_list, self.normal_list, self.rgb_list, self.label_list = [], [], [], []

        for scene in self.scenes:
            scene_path = os.path.join(self.data_root, scene)
            coord = np.load(os.path.join(scene_path, 'coord.npy')).astype(np.float32)
            normal = np.load(os.path.join(scene_path, 'normal.npy')).astype(np.float32)
            rgb = np.load(os.path.join(scene_path, 'color.npy')).astype(np.float32) / 255.0
            label = np.load(os.path.join(scene_path, 'label.npy')).astype(np.int64)

            self.coord_list.append(coord)
            self.normal_list.append(normal)
            self.rgb_list.append(rgb)
            self.label_list.append(label)

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        coord = self.coord_list[idx]
        normal = self.normal_list[idx]
        rgb = self.rgb_list[idx]
        label = self.label_list[idx]

        # 按需采样
        N = coord.shape[0]
        if N >= self.num_points:
            choice = np.random.choice(N, self.num_points, replace=False)
        else:
            choice = np.random.choice(N, self.num_points, replace=True)

        coord = coord[choice]
        normal = normal[choice]
        rgb = rgb[choice]
        label = label[choice]

        feats = []
        if 'xyz' in self.feat_keys:
            feats.append(coord)
        if 'normal' in self.feat_keys:
            feats.append(normal)
        if 'rgb' in self.feat_keys:
            feats.append(rgb)

        feats = np.concatenate(feats, axis=1)  # (num_points, 9)
        return feats, label
