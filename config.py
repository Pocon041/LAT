import os

# ---------- 数据路径 ----------
DIV2K_TRAIN_DIR = r"D:\PNG_Dataset\DIV2K_train_HR"
DIV2K_VAL_DIR = r"D:\PNG_Dataset\DIV2K_valid_HR"
FLICKR2K_DIR = r"D:\Flickr2K\Flickr2K_HR"

GENIMAGE_ROOT = r"D:\Genimage-Tiny"
CHAMELEON_REAL = r"D:\Chameleon\test\0_real"
CHAMELEON_FAKE = r"D:\Chameleon\test\1_fake"

ALL_GENIMAGE_SUBSETS = [
    "imagenet_ai_0419_biggan",
    "imagenet_ai_0419_vqdm",
    "imagenet_ai_0424_sdv5",
    "imagenet_ai_0424_wukong",
    "imagenet_ai_0508_adm",
    "imagenet_glide",
    "imagenet_midjourney",
]

# ---------- 模型 ----------
IMG_SIZE = 256
LATENT_CHANNELS = 128
LATENT_SPATIAL = 16

# ---------- 训练 ----------
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 300
NUM_WORKERS = 4

LAMBDA_L1 = 1.0
LAMBDA_LPIPS = 0.5
LAMBDA_FREQ = 0.1

# ---------- Latent MAE ----------
# 由 train 脚本的命令行参数覆盖
MASK_KEEP_RATIO = 0.7
MASK_ALPHA = 1.0

# ---------- 输出 ----------
EXP_ROOT = r"D:\MAE_Test\experiments"
os.makedirs(EXP_ROOT, exist_ok=True)
