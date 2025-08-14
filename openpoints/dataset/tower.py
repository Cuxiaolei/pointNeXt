import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

__all__ = ['TowerDataset']

class TowerDataset(Dataset):
    def __init__(self, split='train', data_root='data/tower', num_points=8000, feat_keys=['xyz', 'normal', 'rgb'], classes=3, augment=False, **kwargs):
        super().__init__()
        self.split = split
        self.data_root = Path(data_root) / split
        self.num_points = num_points
        self.feat_keys = feat_keys
        self.classes = classes
        self.augment = augment

        assert self.data_root.exists(), f"Data root {self.data_root} not found"
        self.scenes = sorted([d for d in self.data_root.iterdir() if d.is_dir()])
        assert len(self.scenes) > 0, f"No scenes found under {self.data_root}"

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        coord = np.load(scene / "coord.npy").astype(np.float32)
        normal = np.load(scene / "normal.npy").astype(np.float32) if (scene / "normal.npy").exists() else None
        rgb = np.load(scene / "color.npy").astype(np.float32) / 255.0 if (scene / "color.npy").exists() else None
        label = np.load(scene / "label.npy").astype(np.int64)

        N = coord.shape[0]
        if N >= self.num_points:
            choice = np.random.choice(N, self.num_points, replace=False)
        else:
            choice = np.concatenate([np.arange(N), np.random.choice(N, self.num_points - N, replace=True)])
        coord = coord[choice]
        label = label[choice]
        if normal is not None:
            normal = normal[choice]
        if rgb is not None:
            rgb = rgb[choice]

        if self.augment:
            theta = np.random.uniform(-np.pi/12, np.pi/12)
            c, s = np.cos(theta), np.sin(theta)
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            coord = coord @ Rz.T
            if normal is not None:
                normal = normal @ Rz.T
            scale = np.random.uniform(0.9, 1.1)
            coord *= scale
            coord += np.random.normal(0, 0.005, coord.shape)
            if rgb is not None:
                rgb = np.clip(rgb + np.random.normal(0, 0.02, rgb.shape), 0, 1)

        feats = []
        for key in self.feat_keys:
            if key == 'xyz':
                feats.append(coord)
            elif key == 'normal' and normal is not None:
                feats.append(normal)
            elif key == 'rgb' and rgb is not None:
                feats.append(rgb)
            else:
                raise ValueError(f"Unsupported feat_key: {key}")
        feat = np.concatenate(feats, axis=1).astype(np.float32)
        return coord, feat, label
