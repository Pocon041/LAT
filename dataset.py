import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config


def collect_images(folder, exts=(".png", ".jpg", ".jpeg", ".bmp", ".webp")):
    paths = []
    if folder is None or not os.path.isdir(folder):
        return paths
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(exts):
            paths.append(os.path.join(folder, name))
    return paths


def resize_shorter_side(img, target_short=288):
    w, h = img.size
    short_side = min(w, h)
    if short_side == target_short:
        return img
    scale = target_short / short_side
    return img.resize((int(round(w * scale)), int(round(h * scale))), Image.BICUBIC)


def ensure_min_size(img, min_size):
    w, h = img.size
    if w >= min_size and h >= min_size:
        return img
    scale = max(min_size / w, min_size / h) + 1e-3
    return img.resize((int(round(w * scale)), int(round(h * scale))), Image.BICUBIC)


class RealPatchDataset(Dataset):
    def __init__(self, split="train", patches_per_image=None):
        super().__init__()
        self.split = split
        self.is_train = split == "train"
        self.patches_per_image = patches_per_image if patches_per_image is not None else config.PATCHES_PER_IMAGE

        flickr_all = collect_images(config.FLICKR2K_DIR)
        flickr_split = max(0, len(flickr_all) - config.FLICKR2K_VAL_COUNT)
        flickr_train = flickr_all[:flickr_split]
        flickr_val = flickr_all[flickr_split:]

        if self.is_train:
            self.paths = collect_images(config.DIV2K_TRAIN_DIR) + flickr_train
        elif split == "val":
            self.paths = collect_images(config.DIV2K_VAL_DIR) + flickr_val
            self.patches_per_image = 1
        else:
            raise ValueError(f"未知 split: {split}")

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
        self.eval_transform = transforms.Compose([
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths) * self.patches_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        path = self.paths[img_idx]
        img = Image.open(path).convert("RGB")
        if self.is_train:
            img = ensure_min_size(img, config.IMG_SIZE)
            img = self.train_transform(img)
        else:
            img = resize_shorter_side(img, target_short=288)
            img = ensure_min_size(img, config.IMG_SIZE)
            img = self.eval_transform(img)
        return img


class FolderBinaryDataset(Dataset):
    def __init__(self, real_dir, fake_dir, return_path=False):
        super().__init__()
        self.return_path = return_path
        self.items = []
        for path in collect_images(real_dir):
            self.items.append((path, 0))
        for path in collect_images(fake_dir):
            self.items.append((path, 1))
        self.transform = transforms.Compose([
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = resize_shorter_side(img, target_short=288)
        img = ensure_min_size(img, config.IMG_SIZE)
        img = self.transform(img)
        if self.return_path:
            return img, label, path
        return img, label


class ChameleonTestDataset(FolderBinaryDataset):
    def __init__(self, return_path=False):
        super().__init__(
            config.CHAMELEON_REAL_DIR,
            config.CHAMELEON_FAKE_DIR,
            return_path=return_path,
        )


if __name__ == "__main__":
    ds_train = RealPatchDataset(split="train")
    ds_val = RealPatchDataset(split="val")
    ds_test = ChameleonTestDataset()
    print(f"train images={len(ds_train.paths)}, train patches={len(ds_train)}")
    print(f"val images={len(ds_val.paths)}, val patches={len(ds_val)}")
    n_real = sum(1 for _, y in ds_test.items if y == 0)
    n_fake = sum(1 for _, y in ds_test.items if y == 1)
    print(f"test real={n_real}, fake={n_fake}, total={len(ds_test)}")
