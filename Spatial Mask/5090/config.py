import os

# ---------- 数据路径 ----------
DIV2K_TRAIN_DIR = "../autodl-tmp/DIV2K_train_HR"
DIV2K_VAL_DIR = "../autodl-tmp/DIV2K_valid_HR"
FLICKR2K_DIR = "../autodl-tmp/Flickr2K/Flickr2K_HR"
FLICKR2K_VAL_COUNT = 200   # Flickr2K 尾部 200 张划给验证集

# Chameleon zero-shot 测试集
CHAMELEON_REAL_DIR = "../autodl-tmp/Chameleon/test/0_real"
CHAMELEON_FAKE_DIR = "../autodl-tmp/Chameleon/test/1_fake"

# ---------- 模型 ----------
IMG_SIZE = 256
LATENT_CHANNELS = 128
LATENT_SPATIAL = 16
NUM_TOKENS = LATENT_SPATIAL ** 2

# Encoder (ConvNeXt 风格)
ENCODER_DIMS = [64, 128, 256, 128]
ENCODER_DEPTHS = [2, 2, 2, 2]

# Predictor (Transformer)
PRED_NUM_LAYERS = 4
PRED_NUM_HEADS = 8
PRED_DIM_FFN = 512
PRED_DROPOUT = 0.1

# ---------- 掩码 ----------
MASK_RATIOS_MAIN = [0.30, 0.50, 0.60]
MASK_RATIOS_AUX = [0.05, 0.10]
MASK_MAIN_PROB = 0.8

EVAL_MASK_RATIOS = [0.30, 0.40, 0.50, 0.60, 0.70]

# ---------- 5090 + 90GB 内存 优化 ----------
# 5090: 32GB VRAM, ~21760 CUDA cores
# 内存: 90GB, 足够预加载所有训练图像 (~25GB)
TORCH_COMPILE = True        # 启用 torch.compile 加速
CUDNN_BENCHMARK = True      # 固定输入尺寸, cudnn 自动调优
PIN_MEMORY = True
PERSISTENT_WORKERS = True   # 保持 worker 进程存活
PREFETCH_FACTOR = 4         # 每个 worker 预取 4 个 batch
PRELOAD_TO_RAM = True       # 预加载所有图像到内存, 消除磁盘 IO

# ---------- 阶段一训练 ----------
S1_BATCH_SIZE = 64         # 16 -> 64, 5090 32GB 轻松装下
S1_LR = 5e-4               # batch 扩大 4x, lr 线性缩放
S1_EPOCHS = 200
S1_NUM_WORKERS = 8          # 4 -> 8, 喂饱 GPU
S1_LAMBDA_L1 = 1.0
S1_LAMBDA_FREQ = 0.1

# ---------- 阶段二训练 ----------
S2_BATCH_SIZE = 64         # 16 -> 64
S2_LR = 3e-4               # batch 扩大 4x, lr 线性缩放
S2_EPOCHS = 100
S2_NUM_WORKERS = 8
S2_LAMBDA_L1 = 1.0
S2_LAMBDA_COS = 0.5

# ---------- 评估 ----------
EVAL_K = 8
EVAL_K_LIST = [4, 8, 16]
EVAL_BASE_MASK_RATIO = 0.05
EVAL_BASE_RUNS = 3
TAU_PERCENTILE = 5

# ---------- 输出 ----------
EXP_DIR = "./experiments"
os.makedirs(EXP_DIR, exist_ok=True)
