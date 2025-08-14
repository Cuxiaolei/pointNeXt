import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class TowerDataset(Dataset):
    def __init__(self, root, split='train', num_points=8000, augment=True):
        self.root = Path(root) / split
        self.scenes = sorted([d for d in self.root.iterdir() if d.is_dir()])
        self.num_points = num_points
        self.augment = augment and (split == "train")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        d = self.scenes[idx]
        coord = np.load(d / "coord.npy").astype(np.float32)
        color = np.load(d / "color.npy").astype(np.float32) / 255.0
        normal = np.load(d / "normal.npy").astype(np.float32)
        label = np.load(d / "label.npy").astype(np.int64)

        # 固定点数
        N = coord.shape[0]
        if N >= self.num_points:
            choice = np.random.choice(N, self.num_points, replace=False)
        else:
            pad = np.random.choice(N, self.num_points - N, replace=True)
            choice = np.concatenate([np.arange(N), pad])

        coord, color, normal, label = coord[choice], color[choice], normal[choice], label[choice]

        # 数据增强
        if self.augment:
            theta = np.random.uniform(-np.pi/12, np.pi/12)
            c, s = np.cos(theta), np.sin(theta)
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            coord = coord @ Rz.T
            normal = normal @ Rz.T
            coord *= np.random.uniform(0.9, 1.1)
            coord += np.random.normal(0, 0.005, coord.shape)
            color = np.clip(color + np.random.normal(0, 0.02, color.shape), 0, 1)

        feat = np.concatenate([coord, normal, color], axis=1).astype(np.float32)  # 9维
        return coord, feat, label
