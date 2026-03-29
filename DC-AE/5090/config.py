import os

# ---------- 数据路径 ----------
DIV2K_TRAIN_DIR = r"D:\PNG_Dataset\DIV2K_train_HR"
DIV2K_VAL_DIR = r"D:\PNG_Dataset\DIV2K_valid_HR"
FLICKR2K_DIR = r"D:\Flickr2K\Flickr2K_HR"
FLICKR2K_VAL_COUNT = 200   # Flickr2K 尾部 200 张划给验证集

# Chameleon zero-shot 测试集
CHAMELEON_REAL_DIR = r"D:\Chameleon\test\0_real"
CHAMELEON_FAKE_DIR = r"D:\Chameleon\test\1_fake"

# ---------- 模型 ----------
IMG_SIZE = 256
LATENT_CHANNELS = 128
LATENT_SPATIAL = 16        # 16x16 网格
NUM_TOKENS = LATENT_SPATIAL ** 2  # 256

# 结构/细节通道分离 (DC-AE 思想)
STRUCT_CHANNELS = 32       # Z_s: 前32通道 - 宏观结构
DETAIL_CHANNELS = 96       # Z_d: 后96通道 - 高频纹理细节

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
MASK_RATIOS_MAIN = [0.40, 0.60, 0.75]   # 主体分布: 学习空间因果推演
MASK_RATIOS_AUX = [0.05, 0.10]           # 辅助分布: 防止分布偏移
MASK_MAIN_PROB = 0.8   # 80% 概率从主体分布采样

# 测试时 mask ratio sweep
EVAL_MASK_RATIOS = [0.40, 0.50, 0.60, 0.70, 0.75]

# ---------- 5090 性能优化 ----------
CUDNN_BENCHMARK = True     # cudnn.benchmark 自动选择最快算法
PREFETCH_FACTOR = 4        # DataLoader 预取倍数
PIN_MEMORY = True          # 锁页内存加速 GPU 传输
PERSISTENT_WORKERS = True  # 保持 worker 进程存活
COMPILE_MODEL = False      # torch.compile (实验性)
USE_CHANNELS_LAST = True   # channels_last 内存格式，Ampere+ 架构提速

# ---------- 阶段一训练 ----------
S1_BATCH_SIZE = 96         # 5090 32GB 显存，加倍
S1_LR = 6e-4               # batch 翻倍，LR 线性缩放
S1_EPOCHS = 200
S1_NUM_WORKERS = 12        # 90GB 内存, 多 worker 预加载
# L_full-recon = L_pixel-L1 + λ_f * L_freq
S1_LAMBDA_L1 = 1.0
S1_LAMBDA_FREQ = 0.1
# L_struct: 结构通道监督 (Z_s -> X_low)
S1_LAMBDA_STRUCT = 0.5
S1_STRUCT_L1_WEIGHT = 1.0
S1_STRUCT_COS_WEIGHT = 0.2
# L_detail: 细节通道监督 (Z_d -> X_detail)
S1_LAMBDA_DETAIL = 0.1
# X_low 生成: 高斯模糊核大小
S1_BLUR_KERNEL_SIZE = 15
S1_BLUR_SIGMA = 3.0

# ---------- 阶段二训练 ----------
S2_BATCH_SIZE = 128        # Predictor 轻量，可用大 batch
S2_LR = 4e-4               # batch 翻倍，LR 线性缩放
S2_EPOCHS = 100
S2_NUM_WORKERS = 12
# L_mask-struct (前32通道)
S2_LAMBDA_STRUCT = 1.0
S2_STRUCT_L1_WEIGHT = 1.0    # α_1
S2_STRUCT_COS_WEIGHT = 0.3   # α_2
# L_mask-detail (后96通道)
S2_LAMBDA_DETAIL = 0.08
S2_DETAIL_L1_WEIGHT = 1.0    # β_1
S2_DETAIL_COS_WEIGHT = 0.3   # β_2

# ---------- 评估 ----------
EVAL_K = 8             # 默认探测次数
EVAL_K_LIST = [4, 8, 16]
EVAL_BASE_MASK_RATIO = 0.05
EVAL_BASE_RUNS = 3
TAU_PERCENTILE = 5     # tau 取 Err_base 的 5% 分位数
EVAL_BATCH_SIZE = 64   # 评估时批量编码
EVAL_NUM_WORKERS = 8

# ---------- 输出 ----------
EXP_DIR = r"D:\MAE_Test\experiments"
os.makedirs(EXP_DIR, exist_ok=True)
