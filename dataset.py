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


class RealPatchDataset(Dataset):
    """
    DIV2K + Flickr2K 真实图像数据集, 随机裁剪 256x256 patch
    由于原图分辨率远大于 256, 每张图每 epoch 可裁多个 patch
    """
    def __init__(self, patches_per_image=8, split="train"):
        super().__init__()
        self.patches_per_image = patches_per_image

        if split == "train":
            self.paths = (
                _collect_images(config.DIV2K_TRAIN_DIR)
                + _collect_images(config.FLICKR2K_DIR)
            )
        else:
            self.paths = _collect_images(config.DIV2K_VAL_DIR)

        # 随机裁剪 + 数据增强
        self.crop = transforms.RandomCrop(config.IMG_SIZE)
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
        # 验证集不做翻转
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.is_train = (split == "train")

    def __len__(self):
        return len(self.paths) * self.patches_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        img = Image.open(self.paths[img_idx]).convert("RGB")

        # 确保图像足够大, 否则先 resize
        w, h = img.size
        if w < config.IMG_SIZE or h < config.IMG_SIZE:
            scale = max(config.IMG_SIZE / w, config.IMG_SIZE / h) + 0.01
            img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

        img = self.crop(img)
        if self.is_train:
            img = self.transform(img)
        else:
            img = self.val_transform(img)
        return img


class ChameleonTestDataset(Dataset):
    """
    Chameleon zero-shot 测试集
    加载 real + fake, 统一 resize 到 256x256
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
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
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
