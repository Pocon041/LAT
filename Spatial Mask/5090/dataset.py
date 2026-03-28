import os
import io
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

import config as config


def _collect_images(folder, exts=(".png", ".jpg", ".jpeg")):
    """收集文件夹下所有图像路径"""
    paths = []
    if not os.path.isdir(folder):
        return paths
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(exts):
            paths.append(os.path.join(folder, fname))
    return paths


def _resize_shorter_side(img, target_short=288):
    """将图像短边 resize 到 target_short, 保持长宽比"""
    w, h = img.size
    short_side = min(w, h)
    if short_side != target_short:
        scale = target_short / short_side
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    return img


def _preload_images(paths, desc="预加载图像"):
    """将所有图像读入内存, 返回 PIL Image 列表"""
    images = []
    for p in tqdm(paths, desc=desc):
        img = Image.open(p).convert("RGB")
        # 强制加载到内存 (防止惰性解码)
        img.load()
        images.append(img)
    return images


class RealPatchDataset(Dataset):
    """
    5090 优化版: 支持预加载所有图像到内存, 消除磁盘 IO 瓶颈
    train: 随机裁剪 256x256 + 随机翻转
    val: Resize(shorter_side=288) + CenterCrop(256), 完全确定性
    """
    def __init__(self, patches_per_image=8, split="train"):
        super().__init__()
        self.is_train = (split == "train")

        flickr_all = _collect_images(config.FLICKR2K_DIR)
        flickr_split = len(flickr_all) - config.FLICKR2K_VAL_COUNT
        flickr_train = flickr_all[:flickr_split]
        flickr_val = flickr_all[flickr_split:]

        if split == "train":
            self.paths = (
                _collect_images(config.DIV2K_TRAIN_DIR)
                + flickr_train
            )
            self.patches_per_image = patches_per_image
        else:
            self.paths = (
                _collect_images(config.DIV2K_VAL_DIR)
                + flickr_val
            )
            self.patches_per_image = 1

        # 预加载到内存
        self.preloaded = None
        if getattr(config, "PRELOAD_TO_RAM", False):
            self.preloaded = _preload_images(
                self.paths, desc=f"预加载 {split} 集到内存"
            )

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
        self.val_transform = transforms.Compose([
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths) * self.patches_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image

        if self.preloaded is not None:
            img = self.preloaded[img_idx].copy()
        else:
            img = Image.open(self.paths[img_idx]).convert("RGB")

        if self.is_train:
            w, h = img.size
            if w < config.IMG_SIZE or h < config.IMG_SIZE:
                scale = max(config.IMG_SIZE / w, config.IMG_SIZE / h) + 0.01
                img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
            img = self.train_transform(img)
        else:
            img = _resize_shorter_side(img, target_short=288)
            img = self.val_transform(img)
        return img


class ChameleonTestDataset(Dataset):
    """
    5090 优化版 Chameleon 测试集, 支持预加载
    确定性预处理: Resize(shorter_side=288) + CenterCrop(256)
    """
    def __init__(self):
        super().__init__()
        self.items = []  # (path, label)

        for p in _collect_images(config.CHAMELEON_REAL_DIR):
            self.items.append((p, 0))
        for p in _collect_images(config.CHAMELEON_FAKE_DIR):
            self.items.append((p, 1))

        self.preloaded = None
        if getattr(config, "PRELOAD_TO_RAM", False):
            paths = [item[0] for item in self.items]
            self.preloaded = _preload_images(paths, desc="预加载 Chameleon 测试集")

        self.transform = transforms.Compose([
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        if self.preloaded is not None:
            img = self.preloaded[idx].copy()
        else:
            img = Image.open(path).convert("RGB")
        img = _resize_shorter_side(img, target_short=288)
        img = self.transform(img)
        return img, label


if __name__ == "__main__":
    ds_train = RealPatchDataset(split="train")
    ds_val = RealPatchDataset(split="val")
    ds_test = ChameleonTestDataset()

    print(f"训练集: {len(ds_train.paths)} 张原图, "
          f"{len(ds_train)} 个 patch (x{ds_train.patches_per_image})")
    print(f"  预加载: {'是' if ds_train.preloaded else '否'}")
    print(f"验证集: {len(ds_val.paths)} 张原图, "
          f"{len(ds_val)} 个 patch (x{ds_val.patches_per_image})")
    print(f"Chameleon 测试集: {len(ds_test)} 张")
    print(f"  预加载: {'是' if ds_test.preloaded else '否'}")
