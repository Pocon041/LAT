import os

# ---------- 数据路径 ----------
DIV2K_TRAIN_DIR = r"D:\PNG_Dataset\DIV2K_train_HR"
DIV2K_VAL_DIR = r"D:\PNG_Dataset\DIV2K_valid_HR"
FLICKR2K_DIR = r"D:\Flickr2K\Flickr2K_HR"

# Chameleon zero-shot 测试集
CHAMELEON_REAL_DIR = r"D:\Chameleon\test\0_real"
CHAMELEON_FAKE_DIR = r"D:\Chameleon\test\1_fake"

# ---------- 模型 ----------
IMG_SIZE = 256
LATENT_CHANNELS = 128
LATENT_SPATIAL = 16        # 16x16 网格
NUM_TOKENS = LATENT_SPATIAL ** 2  # 256

# Encoder (ConvNeXt 风格)
ENCODER_DIMS = [64, 128, 256, 128]
ENCODER_DEPTHS = [2, 2, 2, 2]

# Predictor (Transformer)
PRED_NUM_LAYERS = 4
PRED_NUM_HEADS = 8
PRED_DIM_FFN = 512
PRED_DROPOUT = 0.1

# ---------- 掩码 ----------
# 训练时 mask ratio 采样: 主体分布 + 辅助分布
MASK_RATIOS_MAIN = [0.30, 0.50, 0.60]
MASK_RATIOS_AUX = [0.05, 0.10]
MASK_MAIN_PROB = 0.8   # 80% 概率从主体分布采样

# 测试时 mask ratio sweep
EVAL_MASK_RATIOS = [0.30, 0.40, 0.50, 0.60, 0.70]

# ---------- 阶段一训练 ----------
S1_BATCH_SIZE = 16
S1_LR = 2e-4
S1_EPOCHS = 200
S1_NUM_WORKERS = 4
S1_LAMBDA_L1 = 1.0
S1_LAMBDA_FREQ = 0.1

# ---------- 阶段二训练 ----------
S2_BATCH_SIZE = 16
S2_LR = 1e-4
S2_EPOCHS = 100
S2_NUM_WORKERS = 4
S2_LAMBDA_L1 = 1.0
S2_LAMBDA_COS = 0.5

# ---------- 评估 ----------
EVAL_K = 8             # 默认探测次数
EVAL_K_LIST = [4, 8, 16]
EVAL_BASE_MASK_RATIO = 0.05
EVAL_BASE_RUNS = 3
TAU_PERCENTILE = 5     # tau 取 Err_base 的 5% 分位数

# ---------- 输出 ----------
EXP_DIR = r"D:\MAE_Test\experiments"
os.makedirs(EXP_DIR, exist_ok=True)
