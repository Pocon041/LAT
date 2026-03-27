import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config


class RealTrainDataset(Dataset):
    """DIV2K train + Flickr2K, 全部 PNG 无损 real 图像"""
    def __init__(self):
        super().__init__()
        self.paths = []
        # DIV2K train
        if os.path.isdir(config.DIV2K_TRAIN_DIR):
            for f in sorted(os.listdir(config.DIV2K_TRAIN_DIR)):
                if f.lower().endswith(".png"):
                    self.paths.append(os.path.join(config.DIV2K_TRAIN_DIR, f))
        # Flickr2K
        if os.path.isdir(config.FLICKR2K_DIR):
            for f in sorted(os.listdir(config.FLICKR2K_DIR)):
                if f.lower().endswith(".png"):
                    self.paths.append(os.path.join(config.FLICKR2K_DIR, f))

        self.transform = transforms.Compose([
            transforms.RandomCrop(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        return img


class RealValDataset(Dataset):
    """DIV2K val, 用于训练时验证和 calib"""
    def __init__(self):
        super().__init__()
        self.paths = []
        if os.path.isdir(config.DIV2K_VAL_DIR):
            for f in sorted(os.listdir(config.DIV2K_VAL_DIR)):
                if f.lower().endswith(".png"):
                    self.paths.append(os.path.join(config.DIV2K_VAL_DIR, f))

        self.transform = transforms.Compose([
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        return img


class GenImageTestDataset(Dataset):
    """GenImage val: DIV2K val 作为 real + GenImage val/ai 作为 fake (全部 PNG)"""
    def __init__(self):
        super().__init__()
        self.items = []  # (path, label, source)
        # real: DIV2K val
        if os.path.isdir(config.DIV2K_VAL_DIR):
            for f in sorted(os.listdir(config.DIV2K_VAL_DIR)):
                if f.lower().endswith(".png"):
                    self.items.append((os.path.join(config.DIV2K_VAL_DIR, f), 0, "DIV2K"))
        # fake: GenImage val/ai
        for sub in config.ALL_GENIMAGE_SUBSETS:
            fake_dir = os.path.join(config.GENIMAGE_ROOT, sub, "val", "ai")
            if os.path.isdir(fake_dir):
                for f in os.listdir(fake_dir):
                    if f.lower().endswith(".png"):
                        self.items.append((os.path.join(fake_dir, f), 1, sub))

        self.transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label, source = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label, source


class ChameleonTestDataset(Dataset):
    """Chameleon zero-shot 测试集"""
    def __init__(self, max_per_class=2000):
        super().__init__()
        self.items = []
        real_files = sorted(os.listdir(config.CHAMELEON_REAL))[:max_per_class]
        for f in real_files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                self.items.append((os.path.join(config.CHAMELEON_REAL, f), 0, "cham_real"))
        fake_files = sorted(os.listdir(config.CHAMELEON_FAKE))[:max_per_class]
        for f in fake_files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                self.items.append((os.path.join(config.CHAMELEON_FAKE, f), 1, "cham_fake"))

        self.transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label, source = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label, source


if __name__ == "__main__":
    ds_train = RealTrainDataset()
    ds_val = RealValDataset()
    ds_gen = GenImageTestDataset()
    ds_cham = ChameleonTestDataset()
    print(f"Real train (DIV2K+Flickr2K): {len(ds_train)}")
    print(f"Real val (DIV2K val): {len(ds_val)}")
    n_real = sum(1 for _, l, _ in ds_gen.items if l == 0)
    n_fake = sum(1 for _, l, _ in ds_gen.items if l == 1)
    print(f"GenImage test: real={n_real}, fake={n_fake}, total={len(ds_gen)}")
    n_real = sum(1 for _, l, _ in ds_cham.items if l == 0)
    n_fake = sum(1 for _, l, _ in ds_cham.items if l == 1)
    print(f"Chameleon test: real={n_real}, fake={n_fake}, total={len(ds_cham)}")
