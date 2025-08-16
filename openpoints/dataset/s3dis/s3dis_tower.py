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
                 data_root: str = '/root/autodl-tmp/raw',
                 test_area: int = 5,
                 voxel_size: float = 0.04,
                 voxel_max=None,
                 split: str = 'train',
                 transform=None,
                 loop: int = 1,
                 presample: bool = False,
                 variable: bool = False,
                 shuffle: bool = True,
                 ):
        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max, self.loop = \
            split, voxel_size, transform, voxel_max, loop
        self.presample = presample
        self.variable = variable
        self.shuffle = shuffle

        # 修改: 使用 merged/ 下的场景文件
        merged_root = os.path.join(data_root, 'merged')
        self.merged_root = merged_root

        # 修改: 读取 train_scenes.txt / val_scenes.txt / test_scenes.txt
        split_file = os.path.join(data_root, f"{split}_scenes.txt")
        assert os.path.exists(split_file), f"{split_file} 不存在，请先运行数据预处理生成"
        with open(split_file, "r") as f:
            self.data_list = [line.strip() for line in f.readlines() if line.strip()]

        processed_root = os.path.join(data_root, 'processed')
        filename = os.path.join(
            processed_root, f's3dis_{split}_{voxel_size:.3f}_{str(voxel_max)}.pkl')

        if presample and not os.path.exists(filename):
            np.random.seed(0)
            self.data = []
            for item in tqdm(self.data_list, desc=f'Loading S3DISTower {split} split'):
                data_path = os.path.join(data_root, item)  # 修改: 直接用相对路径
                cdata = np.load(data_path).astype(np.float32)
                cdata[:, :3] -= np.min(cdata[:, :3], 0)
                if voxel_size:
                    coord, feat, label = cdata[:, 0:3], cdata[:, 3:6], cdata[:, 6:7]
                    uniq_idx = voxelize(coord, voxel_size)
                    coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
                    cdata = np.hstack((coord, feat, label))
                self.data.append(cdata)
            npoints = np.array([len(data) for data in self.data])
            logging.info('split: %s, median npoints %.1f, avg num points %.1f, std %.1f' %
                         (self.split, np.median(npoints), np.average(npoints), np.std(npoints)))
            os.makedirs(processed_root, exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(self.data, f)
                print(f"{filename} saved successfully")
        elif presample:
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)
                print(f"{filename} load successfully")

        self.data_idx = np.arange(len(self.data_list))
        assert len(self.data_idx) > 0
        logging.info(f"\nTotally {len(self.data_idx)} samples in {split} set")

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        if self.presample:
            coord, feat, label = np.split(self.data[data_idx], [3, 6], axis=1)
        else:
            data_path = os.path.join(self.raw_root, self.data_list[data_idx] + '.npy')
            cdata = np.load(data_path).astype(np.float32)
            cdata[:, :3] -= np.min(cdata[:, :3], 0)
            coord, feat, label = cdata[:, :3], cdata[:, 3:6], cdata[:, 6:7]
            coord, feat, label = crop_pc(
                coord, feat, label, self.split, self.voxel_size, self.voxel_max,
                downsample=not self.presample, variable=self.variable, shuffle=self.shuffle)

        # 获取样本名称（方便调试输出）
        sample_name = self.data_names[idx] if hasattr(self, "data_names") else f"idx_{idx}"

        # 保证格式正确
        coord, feat, label = _ensure_shapes(coord, feat, label)

        # ===== 调试日志：检查特征缺失 =====
        if feat is None or feat.shape[1] < 3:
            print(f"[Debug][{self.split}] {sample_name} 缺失 RGB 特征: "
                  f"feat shape={None if feat is None else feat.shape}")
        else:
            # 检查每个通道是否恒为0或恒为128（可能是补齐值）
            for ch_idx, ch_name in enumerate(["R", "G", "B"]):
                unique_vals = np.unique(feat[:, ch_idx])
                if len(unique_vals) == 1:
                    print(f"[Debug][{self.split}] {sample_name} {ch_name} 通道恒为 {unique_vals[0]}，可能缺失原始值")
        # =================================



        coord, feat, mask = _sanitize_numeric(coord, feat)
        if label is not None:
            label = label[mask]
        voxel_max = getattr(self, 'voxel_max', None)
        sample_name = self.data_list[data_idx] if hasattr(self, 'data_list') else f"idx_{data_idx}"
        coord, feat, label = _limit_points(coord, feat, label, voxel_max, self.split, sample_name)

        # ====== 新增安全检查 ======
        if coord.shape[0] < 1:
            raise ValueError(f"[{self.split}] sample {sample_name} has no points after processing")
        if not np.isfinite(coord).all() or not np.isfinite(feat).all():
            raise ValueError(f"[{self.split}] sample {sample_name} contains NaN/Inf values")
        if feat.shape[1] + coord.shape[1] != 6:
            raise ValueError(f"[{self.split}] sample {sample_name} feature dim mismatch: coord({coord.shape[1]})+feat({feat.shape[1]}) != 6")
        # =========================

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
