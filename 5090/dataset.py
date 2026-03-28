import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading

import config


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


def _load_single_image(args):
    """加载单张图像 (用于多线程)"""
    path, resize_short = args
    try:
        img = Image.open(path).convert("RGB")
        if resize_short:
            img = _resize_shorter_side(img, target_short=resize_short)
        return img
    except Exception as e:
        print(f"加载失败: {path}, {e}")
        return None


def _preload_images(paths, resize_short=None, num_threads=16, desc="预加载"):
    """多线程预加载图像到内存"""
    print(f"[RAM预加载] {desc}: {len(paths)} 张图像, {num_threads} 线程")
    
    args_list = [(p, resize_short) for p in paths]
    images = [None] * len(paths)
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(tqdm(
            executor.map(_load_single_image, args_list),
            total=len(paths),
            desc=desc
        ))
    
    for i, img in enumerate(results):
        images[i] = img
    
    valid_count = sum(1 for img in images if img is not None)
    print(f"[RAM预加载] 完成: {valid_count}/{len(paths)} 张有效")
    return images


class RealPatchDataset(Dataset):
    """
    DIV2K + Flickr2K 真实图像数据集 (支持 RAM 预加载)
    preload=True: 启动时加载所有图像到内存 (需要大内存)
    """
    def __init__(self, patches_per_image=8, split="train", preload=True, num_threads=16):
        super().__init__()
        self.is_train = (split == "train")
        self.preload = preload

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
        if preload:
            resize_short = None if self.is_train else 288
            self.images = _preload_images(
                self.paths, 
                resize_short=resize_short,
                num_threads=num_threads,
                desc=f"预加载 {split}"
            )
        else:
            self.images = None

        # 训练: 随机裁剪 + 数据增强
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
        # 验证: 确定性裁剪
        self.val_transform = transforms.Compose([
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths) * self.patches_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        
        # 从内存或磁盘获取图像
        if self.preload and self.images[img_idx] is not None:
            img = self.images[img_idx].copy()  # 复制避免修改原图
        else:
            img = Image.open(self.paths[img_idx]).convert("RGB")

        if self.is_train:
            w, h = img.size
            if w < config.IMG_SIZE or h < config.IMG_SIZE:
                scale = max(config.IMG_SIZE / w, config.IMG_SIZE / h) + 0.01
                img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
            img = self.train_transform(img)
        else:
            if not self.preload:
                img = _resize_shorter_side(img, target_short=288)
            img = self.val_transform(img)
        return img


class ChameleonTestDataset(Dataset):
    """
    Chameleon zero-shot 测试集 (支持 RAM 预加载)
    """
    def __init__(self, preload=True, num_threads=16):
        super().__init__()
        self.items = []
        self.preload = preload

        for p in _collect_images(config.CHAMELEON_REAL_DIR):
            self.items.append((p, 0))
        for p in _collect_images(config.CHAMELEON_FAKE_DIR):
            self.items.append((p, 1))

        # 预加载到内存 (resize 到短边 288)
        if preload:
            paths = [p for p, _ in self.items]
            self.images = _preload_images(
                paths, 
                resize_short=288,
                num_threads=num_threads,
                desc="预加载 Chameleon"
            )
        else:
            self.images = None

        self.transform = transforms.Compose([
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        
        if self.preload and self.images[idx] is not None:
            img = self.images[idx]
        else:
            img = Image.open(path).convert("RGB")
            img = _resize_shorter_side(img, target_short=288)
        
        img = self.transform(img)
        return img, label


if __name__ == "__main__":
    print("测试 RAM 预加载...")
    ds_train = RealPatchDataset(split="train", preload=True, num_threads=16)
    ds_val = RealPatchDataset(split="val", preload=True, num_threads=16)
    ds_test = ChameleonTestDataset(preload=True, num_threads=16)

    print(f"\n训练集: {len(ds_train.paths)} 张原图, {len(ds_train)} 个 patch")
    print(f"验证集: {len(ds_val.paths)} 张原图, {len(ds_val)} 个 patch")
    n_real = sum(1 for _, l in ds_test.items if l == 0)
    n_fake = sum(1 for _, l in ds_test.items if l == 1)
    print(f"Chameleon: real={n_real}, fake={n_fake}, 总计={len(ds_test)}")
    
    # 测试访问速度
    import time
    t0 = time.time()
    for i in range(100):
        _ = ds_train[i]
    print(f"\n100 次访问耗时: {time.time() - t0:.3f}s")
