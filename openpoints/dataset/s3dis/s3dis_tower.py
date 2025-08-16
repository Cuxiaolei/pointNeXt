import os
import pickle
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from ..data_util import crop_pc, voxelize
from ..build import DATASETS


def _ensure_2d(arr, name):
    """确保数组是二维的"""
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    return arr


def _sanitize_numeric(coord, feat):
    """去掉 NaN/Inf 并填 0"""
    mask_c = np.isfinite(coord).all(axis=1)
    mask_f = np.isfinite(feat).all(axis=1) if feat is not None else np.ones(len(coord), dtype=bool)
    mask = mask_c & mask_f
    if not mask.all():
        coord = coord[mask]
        feat = feat[mask] if feat is not None else None
    coord = np.nan_to_num(coord, nan=0.0, posinf=0.0, neginf=0.0)
    if feat is not None:
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return coord, feat, mask


def _limit_points(coord, feat, label, voxel_max, split, sample_name):
    """限制点数到 voxel_max"""
    if voxel_max is None or voxel_max <= 0:
        return coord, feat, label
    n = coord.shape[0]
    if n > voxel_max:
        choice = np.random.choice(n, voxel_max, replace=False)
        coord = coord[choice]
        if feat is not None:
            feat = feat[choice]
        if label is not None:
            label = label[choice]
        if split != 'train':
            print(f"[{split}] limit points: {sample_name} {n} -> {voxel_max}")
    return coord, feat, label


def _ensure_shapes(coord, feat, label):
    """确保列数符合 XYZ+RGB+Label 的要求"""
    coord = _ensure_2d(coord, "coord")
    if coord.shape[1] != 3:
        raise ValueError(f"coord must have 3 columns, got {coord.shape[1]}")
    if feat is not None:
        feat = _ensure_2d(feat, "feat")
        if feat.shape[1] < 3:
            raise ValueError(f"feat must have at least 3 columns (rgb), got {feat.shape[1]}")
    if label is not None:
        label = _ensure_2d(label, "label")
        if label.shape[1] != 1:
            raise ValueError(f"label must have 1 column, got {label.shape[1]}")
    return coord, feat, label


@DATASETS.register_module()
class S3DISTower(Dataset):
    classes = ['Tower_Insulator', 'Background', 'Conductor']
    num_classes = 3
    num_per_class = np.array([0, 0, 0], dtype=np.int32)
    class2color = {
        'Tower_Insulator': [0, 255, 0],
        'Background': [0, 0, 255],
        'Conductor': [255, 255, 0]
    }
    cmap = [*class2color.values()]
    gravity_dim = 2

    def __init__(self,
                 data_root: str = '/root/data/data_s3dis_pointNeXt',
                 test_area: int = 5,
                 voxel_size: float = 0.04,
                 voxel_max=None,
                 split: str = 'train',
                 transform=None,
                 loop: int = 1,
                 presample: bool = True,
                 variable: bool = False,
                 shuffle: bool = True,
                 ):
        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max, self.loop = \
            split, voxel_size, transform, voxel_max, loop
        self.presample = presample
        self.variable = variable
        self.shuffle = shuffle

        # === 目录结构：data_root 下只有 merged 与 processed 同级 ===
        # merged/: 场景级 .npy（Area_*.npy）
        # processed/: 各 split 合并缓存的 .pkl
        merged_root = os.path.join(data_root, 'merged')
        processed_root = os.path.join(data_root, 'processed')
        os.makedirs(processed_root, exist_ok=True)
        self.merged_root = merged_root

        # === 列出 merged 下的场景文件 ===
        if not os.path.isdir(merged_root):
            raise FileNotFoundError(f"merged directory not found: {merged_root}")

        all_files = sorted(os.listdir(merged_root))
        # 仅保留 Area_*.npy，并去掉扩展名，后续按 data_list + '.npy' 读取
        data_list = [f[:-4] for f in all_files if f.startswith('Area_') and f.endswith('.npy')]

        # === 根据 split 选择场景 ===
        list_file = os.path.join(data_root, f"{split}_scenes.txt")
        if not os.path.isfile(list_file):
            raise FileNotFoundError(f"Missing split file: {list_file}")

        logging.info(f"[{split}] Using scene list file: {list_file}")

        self.data_list = []
        with open(list_file, "r") as f:
            for line in f.readlines():
                name = line.strip()
                if not name:
                    continue
                # 去掉目录前缀
                if name.startswith("merged/"):
                    name = name.split("/")[-1]
                # 去掉扩展名
                if name.endswith(".npy"):
                    name = name[:-4]
                self.data_list.append(name)

        logging.info(
            f"[{split}] Found {len(self.data_list)} scenes: {self.data_list[:5]}{'...' if len(self.data_list) > 5 else ''}")

        # === 预采样缓存 pkl 的路径（按 split 单独保存） ===
        self.pkl_path = os.path.join(
            processed_root, f's3dis_{split}_{voxel_size:.3f}_{str(voxel_max)}.pkl'
        )
        logging.info(f"[{split}] Cache pkl path: {self.pkl_path}")

        # === presample: 合并该 split 的所有场景为一个 pkl（list，每个元素=一个场景大点云） ===
        if self.presample and not os.path.exists(self.pkl_path):
            np.random.seed(0)
            self.data = []
            for item in tqdm(self.data_list, desc=f'Loading S3DISTower {split} split'):
                npy_path = os.path.join(merged_root, item + '.npy')
                if not os.path.isfile(npy_path):
                    raise FileNotFoundError(f"missing scene npy: {npy_path}")
                cdata = np.load(npy_path).astype(np.float32)
                # 对齐坐标原点到该场景最小值（与你原逻辑一致）
                cdata[:, :3] -= np.min(cdata[:, :3], 0)
                # 体素下采样（与原逻辑一致）
                if voxel_size:
                    coord, feat, label = cdata[:, 0:3], cdata[:, 3:6], cdata[:, 6:7]
                    uniq_idx = voxelize(coord, voxel_size)
                    coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
                    cdata = np.hstack((coord, feat, label))
                self.data.append(cdata)

            npoints = np.array([len(arr) for arr in self.data])
            logging.info('split: %s, median npoints %.1f, avg num points %.1f, std %.1f' %
                         (self.split, np.median(npoints), np.average(npoints), np.std(npoints)))
            with open(self.pkl_path, 'wb') as f:
                pickle.dump(self.data, f)
                print(f"{self.pkl_path} saved successfully")
        elif self.presample:
            with open(self.pkl_path, 'rb') as f:
                self.data = pickle.load(f)
                print(f"{self.pkl_path} load successfully")

        # === 索引与长度 ===
        self.data_idx = np.arange(len(self.data_list))
        assert len(self.data_idx) > 0, f"No samples found for split={split}. Check merged/ and test_area."

        logging.info(f"\nTotally {len(self.data_idx)} samples in {split} set")

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]

        if self.presample:
            # 从 pkl 中取出该场景
            coord, feat, label = np.split(self.data[data_idx], [3, 6], axis=1)

        else:
            # 直接从 merged 读取该场景的 .npy
            npy_path = os.path.join(self.merged_root, self.data_list[data_idx] + '.npy')
            if not os.path.isfile(npy_path):
                raise FileNotFoundError(f"missing scene npy: {npy_path}")
            cdata = np.load(npy_path).astype(np.float32)
            cdata[:, :3] -= np.min(cdata[:, :3], 0)
            coord, feat, label = cdata[:, :3], cdata[:, 3:6], cdata[:, 6:7]
            coord, feat, label = crop_pc(
                coord, feat, label, self.split, self.voxel_size, self.voxel_max,
                downsample=not self.presample, variable=self.variable, shuffle=self.shuffle)

        # 样本名，便于调试
        sample_name = self.data_list[data_idx]

        # 形状与数值健壮性检查
        coord, feat, label = _ensure_shapes(coord, feat, label)

        # （可选）RGB 缺失排查输出
        if feat is None or feat.shape[1] < 3:
            print(f"[Debug][{self.split}] {sample_name} 缺失 RGB 特征: "
                  f"feat shape={None if feat is None else feat.shape}")
        else:
            for ch_idx, ch_name in enumerate(["R", "G", "B"]):
                unique_vals = np.unique(feat[:, ch_idx])
                if len(unique_vals) == 1:
                    print(f"[Debug][{self.split}] {sample_name} {ch_name} 通道恒为 {unique_vals[0]}，可能缺失原始值")

        coord, feat, mask = _sanitize_numeric(coord, feat)
        if label is not None:
            label = label[mask]
        voxel_max = getattr(self, 'voxel_max', None)
        coord, feat, label = _limit_points(coord, feat, label, voxel_max, self.split, sample_name)

        # 安全检查
        if coord.shape[0] < 1:
            raise ValueError(f"[{self.split}] sample {sample_name} has no points after processing")
        if not np.isfinite(coord).all() or not np.isfinite(feat).all():
            raise ValueError(f"[{self.split}] sample {sample_name} contains NaN/Inf values")
        if feat.shape[1] + coord.shape[1] != 6:
            raise ValueError(
                f"[{self.split}] sample {sample_name} feature dim mismatch: "
                f"coord({coord.shape[1]})+feat({feat.shape[1]}) != 6"
            )

        # --- 归一化 RGB ---
        feat = feat.astype(np.float32)
        if feat.max() > 1.0:  # 说明是 0~255 范围
            feat = feat / 255.0
        feat = np.clip(feat, 0.0, 1.0)


        full_feat = np.hstack([coord, feat])
        label = label.squeeze(-1).astype(np.long)


        data = {'pos': coord, 'x': full_feat, 'y': label}
        if self.transform is not None:
            data = self.transform(data)
        if 'heights' not in data.keys():
            data['heights'] = torch.from_numpy(coord[:, self.gravity_dim:self.gravity_dim+1].astype(np.float32))
        return data

    def __len__(self):
        return len(self.data_idx) * self.loop
