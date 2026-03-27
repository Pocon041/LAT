import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

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


class RealPatchDataset(Dataset):
    """
    DIV2K + Flickr2K 真实图像数据集
    train: 随机裁剪 256x256 + 随机翻转
    val: Resize(shorter_side=288) + CenterCrop(256), 完全确定性
    """
    def __init__(self, patches_per_image=8, split="train"):
        super().__init__()
        self.is_train = (split == "train")

        if split == "train":
            self.paths = (
                _collect_images(config.DIV2K_TRAIN_DIR)
                + _collect_images(config.FLICKR2K_DIR)
            )
            self.patches_per_image = patches_per_image
        else:
            self.paths = _collect_images(config.DIV2K_VAL_DIR)
            self.patches_per_image = 1  # val 每张图只出一个确定性 patch

        # 训练: 随机裁剪 + 数据增强
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
        # 验证: 确定性裁剪 (短边 resize + 中心裁剪)
        self.val_transform = transforms.Compose([
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths) * self.patches_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        img = Image.open(self.paths[img_idx]).convert("RGB")

        if self.is_train:
            # 确保图像足够大
            w, h = img.size
            if w < config.IMG_SIZE or h < config.IMG_SIZE:
                scale = max(config.IMG_SIZE / w, config.IMG_SIZE / h) + 0.01
                img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
            img = self.train_transform(img)
        else:
            # 确定性: 短边 resize 到 288 + CenterCrop 256
            img = _resize_shorter_side(img, target_short=288)
            img = self.val_transform(img)
        return img


class ChameleonTestDataset(Dataset):
    """
    Chameleon zero-shot 测试集
    确定性预处理: Resize(shorter_side=288) + CenterCrop(256)
    与训练验证集保持同一 pipeline, 不做拉伸变形
    返回 (img, label): label=0 real, label=1 fake
    """
    def __init__(self):
        super().__init__()
        self.items = []  # (path, label)

        for p in _collect_images(config.CHAMELEON_REAL_DIR):
            self.items.append((p, 0))
        for p in _collect_images(config.CHAMELEON_FAKE_DIR):
            self.items.append((p, 1))

        self.transform = transforms.Compose([
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = _resize_shorter_side(img, target_short=288)
        img = self.transform(img)
        return img, label


if __name__ == "__main__":
    ds_train = RealPatchDataset(split="train")
    ds_val = RealPatchDataset(split="val", patches_per_image=4)
    ds_test = ChameleonTestDataset()

    print(f"训练集: {len(ds_train.paths)} 张原图, "
          f"{len(ds_train)} 个 patch (x{ds_train.patches_per_image})")
    print(f"验证集: {len(ds_val.paths)} 张原图, "
          f"{len(ds_val)} 个 patch (x{ds_val.patches_per_image})")
    n_real = sum(1 for _, l in ds_test.items if l == 0)
    n_fake = sum(1 for _, l in ds_test.items if l == 1)
    print(f"Chameleon 测试集: real={n_real}, fake={n_fake}, 总计={len(ds_test)}")
