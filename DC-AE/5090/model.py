import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ============================================================
#  ConvNeXt 风格 Encoder: 256x256x3 -> 16x16x128
# ============================================================

class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block: depthwise conv -> LayerNorm -> 1x1 -> GELU -> 1x1"""
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim * expansion, dim)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # B C H W -> B H W C
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # B H W C -> B C H W
        return x + shortcut


class Encoder(nn.Module):
    """
    ConvNeXt 风格编码器
    4 个下采样阶段: 256->128->64->32->16
    通道: 3 -> dims[0] -> dims[1] -> dims[2] -> dims[3]
    最后投射到 latent_channels
    """
    def __init__(self, in_channels=3, dims=None, depths=None, latent_channels=128):
        super().__init__()
        if dims is None:
            dims = config.ENCODER_DIMS
        if depths is None:
            depths = config.ENCODER_DEPTHS

        self.stages = nn.ModuleList()

        # 第一阶段: stem (4x4 stride-2 conv + LN)
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], 4, stride=2, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
        )
        blocks = [ConvNeXtBlock(dims[0]) for _ in range(depths[0])]
        self.stages.append(nn.Sequential(stem, *blocks))

        # 后续阶段: stride-2 下采样 + ConvNeXt blocks
        for i in range(1, len(dims)):
            downsample = nn.Sequential(
                nn.Conv2d(dims[i - 1], dims[i], 2, stride=2),
                nn.BatchNorm2d(dims[i]),
            )
            blocks = [ConvNeXtBlock(dims[i]) for _ in range(depths[i])]
            self.stages.append(nn.Sequential(downsample, *blocks))

        # 投射到 latent_channels
        self.proj = nn.Identity()
        if dims[-1] != latent_channels:
            self.proj = nn.Conv2d(dims[-1], latent_channels, 1)

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return self.proj(x)


# ============================================================
#  轻型解码器 D: 16x16x128 -> 256x256x3
# ============================================================

class Decoder(nn.Module):
    """
    轻型解码器, 4 层上采样: 16->32->64->128->256
    使用 PixelShuffle + Conv, 容量受控
    """
    def __init__(self, latent_channels=128, ch_list=None, out_channels=3):
        super().__init__()
        if ch_list is None:
            ch_list = [96, 64, 32]

        layers = []
        c_in = latent_channels
        for c_out in ch_list:
            # PixelShuffle x2: 需要 c_out*4 个通道
            layers.append(nn.Conv2d(c_in, c_out * 4, 3, padding=1))
            layers.append(nn.PixelShuffle(2))
            layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.GELU())
            layers.append(nn.Conv2d(c_out, c_out, 3, padding=1))
            layers.append(nn.GELU())
            c_in = c_out

        # 最后一层上采样到 256
        layers.append(nn.Conv2d(c_in, out_channels * 4, 3, padding=1))
        layers.append(nn.PixelShuffle(2))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


# ============================================================
#  辅助头: H_s (结构) 和 H_d (细节)
# ============================================================

class StructHead(nn.Module):
    """
    结构辅助头 H_s: Z_s (16x16x32) -> X_low (256x256x3)
    输入只有前 32 通道, 重建低频结构图
    """
    def __init__(self, struct_channels=32, out_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(struct_channels, 64 * 4, 3, padding=1),
            nn.PixelShuffle(2),  # 16->32
            nn.GELU(),
            nn.Conv2d(64, 64 * 4, 3, padding=1),
            nn.PixelShuffle(2),  # 32->64
            nn.GELU(),
            nn.Conv2d(64, 32 * 4, 3, padding=1),
            nn.PixelShuffle(2),  # 64->128
            nn.GELU(),
            nn.Conv2d(32, out_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),  # 128->256
            nn.Sigmoid(),
        )

    def forward(self, z_s):
        return self.net(z_s)


class DetailHead(nn.Module):
    """
    细节辅助头 H_d: Z_d (16x16x96) -> X_detail (256x256x3)
    输入只有后 96 通道, 重建细节残差 (X - X_low)
    注意: 输出不用 sigmoid, 直接输出实值残差
    """
    def __init__(self, detail_channels=96, out_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(detail_channels, 96 * 4, 3, padding=1),
            nn.PixelShuffle(2),  # 16->32
            nn.GELU(),
            nn.Conv2d(96, 64 * 4, 3, padding=1),
            nn.PixelShuffle(2),  # 32->64
            nn.GELU(),
            nn.Conv2d(64, 32 * 4, 3, padding=1),
            nn.PixelShuffle(2),  # 64->128
            nn.GELU(),
            nn.Conv2d(32, out_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),  # 128->256
            # 不用 sigmoid, 直接输出实值残差
        )

    def forward(self, z_d):
        return self.net(z_d)


# ============================================================
#  2D 正弦-余弦位置编码
# ============================================================

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    生成 2D 正弦-余弦位置编码
    embed_dim: 每个 token 的位置向量维度 (= latent_channels)
    grid_size: 空间网格大小 (16)
    返回: [grid_size*grid_size, embed_dim]
    """
    half_dim = embed_dim // 2
    grid_y = torch.arange(grid_size, dtype=torch.float32)
    grid_x = torch.arange(grid_size, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
    grid_y = grid_y.reshape(-1)  # [N]
    grid_x = grid_x.reshape(-1)  # [N]

    # 每个坐标分别编码 half_dim 维
    quarter = half_dim // 2
    omega = 1.0 / (10000 ** (torch.arange(quarter, dtype=torch.float32) / quarter))

    # y 方向
    out_y = grid_y.unsqueeze(1) * omega.unsqueeze(0)  # [N, quarter]
    emb_y = torch.cat([out_y.sin(), out_y.cos()], dim=1)  # [N, half_dim]

    # x 方向
    out_x = grid_x.unsqueeze(1) * omega.unsqueeze(0)
    emb_x = torch.cat([out_x.sin(), out_x.cos()], dim=1)

    # 拼接
    pos_embed = torch.cat([emb_y, emb_x], dim=1)  # [N, embed_dim]
    return pos_embed


# ============================================================
#  掩码模块 f
# ============================================================

def random_token_mask(num_tokens, mask_ratio):
    """
    随机 token 掩码
    返回: (mask [num_tokens], actual_ratio)
    """
    num_mask = int(num_tokens * mask_ratio)
    num_mask = max(1, min(num_mask, num_tokens - 1))
    perm = torch.randperm(num_tokens)
    mask = torch.zeros(num_tokens, dtype=torch.bool)
    mask[perm[:num_mask]] = True
    return mask, num_mask / num_tokens


def block_mask(grid_size, mask_ratio):
    """
    连续块掩码: 在 grid_size x grid_size 网格上随机选一个矩形块
    返回: (mask [grid_size*grid_size], actual_ratio)
    """
    num_tokens = grid_size * grid_size
    target_masked = int(num_tokens * mask_ratio)
    target_masked = max(1, min(target_masked, num_tokens - 1))

    # 选择合适的块大小
    area = target_masked
    aspect = random.uniform(0.5, 2.0)
    h = max(1, min(int(round(math.sqrt(area * aspect))), grid_size))
    w = max(1, min(int(round(area / h)), grid_size))

    # 随机选择左上角
    top = random.randint(0, grid_size - h)
    left = random.randint(0, grid_size - w)

    mask = torch.zeros(grid_size, grid_size, dtype=torch.bool)
    mask[top:top + h, left:left + w] = True
    flat = mask.reshape(-1)
    actual = flat.sum().item() / num_tokens
    return flat, actual


def sample_mask(grid_size, mask_ratio, mask_type="random"):
    """
    统一的掩码采样接口
    返回: (mask [N], actual_ratio)
    """
    num_tokens = grid_size * grid_size
    if mask_type == "random":
        return random_token_mask(num_tokens, mask_ratio)
    elif mask_type == "block":
        return block_mask(grid_size, mask_ratio)
    else:
        raise ValueError(f"未知的 mask 类型: {mask_type}")


def sample_train_mask(grid_size):
    """
    训练时的掩码采样策略:
    - 80% 概率从主体分布 [0.3, 0.5, 0.6] 采样
    - 20% 概率从辅助分布 [0.05, 0.10] 采样
    - mask 类型: random 和 block 各 50%
    返回: (mask, actual_ratio, mask_type)
    """
    if random.random() < config.MASK_MAIN_PROB:
        ratio = random.choice(config.MASK_RATIOS_MAIN)
    else:
        ratio = random.choice(config.MASK_RATIOS_AUX)

    mask_type = random.choice(["random", "block"])
    mask, actual_ratio = sample_mask(grid_size, ratio, mask_type)
    return mask, actual_ratio, mask_type


def batch_sample_train_masks(batch_size, grid_size, device):
    """
    批量采样训练 mask，random mask 全量向量化，block mask 仍需循环。
    返回: masks [B, N]
    """
    num_tokens = grid_size * grid_size
    ratios = [
        random.choice(config.MASK_RATIOS_MAIN) if random.random() < config.MASK_MAIN_PROB
        else random.choice(config.MASK_RATIOS_AUX)
        for _ in range(batch_size)
    ]
    type_choices = random.choices(["random", "block"], k=batch_size)
    random_idx = [i for i, t in enumerate(type_choices) if t == "random"]
    block_idx  = [i for i, t in enumerate(type_choices) if t == "block"]

    masks = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)

    if random_idx:
        # 批量向量化: 一次 rand + argsort
        sub = len(random_idx)
        num_masks = torch.tensor([max(1, min(int(num_tokens * ratios[i]), num_tokens - 1))
                                  for i in random_idx], device=device)
        noise = torch.rand(sub, num_tokens, device=device)
        ids = torch.argsort(noise, dim=1)
        for k, i in enumerate(random_idx):
            masks[i, ids[k, :num_masks[k]]] = True

    for i in block_idx:
        m, _ = block_mask(grid_size, ratios[i])
        masks[i] = m.to(device)

    return masks


def batch_sample_masks(batch_size, grid_size, mask_ratio, mask_type, device):
    """
    验证时批量采样固定 ratio/type 的 mask。
    random mask 全量向量化。
    返回: masks [B, N]
    """
    num_tokens = grid_size * grid_size
    if mask_type == "random":
        num_mask = max(1, min(int(num_tokens * mask_ratio), num_tokens - 1))
        noise = torch.rand(batch_size, num_tokens, device=device)
        ids = torch.argsort(noise, dim=1)
        masks = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)
        masks.scatter_(1, ids[:, :num_mask], True)
        return masks
    masks = torch.stack([
        sample_mask(grid_size, mask_ratio, mask_type)[0] for _ in range(batch_size)
    ]).to(device)
    return masks


# ============================================================
#  补全网络 P (Transformer Encoder)
# ============================================================

class Predictor(nn.Module):
    """
    Transformer Encoder 补全网络
    输入: masked latent tokens + 2D 位置编码
    输出: 补全后的 latent tokens
    """
    def __init__(
        self,
        embed_dim=128,
        num_layers=4,
        num_heads=8,
        dim_ffn=512,
        dropout=0.1,
        grid_size=16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size

        # 可学习 [MASK] token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # 2D 位置编码 (固定, 不可学习)
        pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0))  # [1, N, D]

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, z, mask):
        """
        z: [B, C, H, W] latent 特征图
        mask: [B, N] bool, True 表示被 mask
        返回: z_hat [B, C, H, W], 物理替换后的完整 latent
        """
        B, C, H, W = z.shape
        N = H * W

        # 展平为 token 序列 [B, N, C]
        tokens = z.flatten(2).permute(0, 2, 1)  # [B, N, C]

        # 构造输入: 被 mask 的位置替换为 mask_token
        mask_expanded = mask.unsqueeze(-1).float()  # [B, N, 1]
        inp = tokens * (1 - mask_expanded) + self.mask_token * mask_expanded

        # 加入 2D 位置编码
        inp = inp + self.pos_embed

        # Transformer 推理
        out = self.transformer(inp)
        out = self.norm(out)

        # 物理替换: 可见区域保留原始值, 被 mask 区域用预测值
        z_pred = tokens * (1 - mask_expanded) + out * mask_expanded

        # 恢复空间形状 [B, C, H, W]
        z_hat = z_pred.permute(0, 2, 1).reshape(B, C, H, W)
        return z_hat


# ============================================================
#  完整模型
# ============================================================

class LatentMAE(nn.Module):
    """
    两阶段 Latent MAE 模型 (Structure-Aware)
    阶段一: E + D + H_s + H_d (AE 预训练, 结构/细节解耦)
    阶段二: P (masked latent completion), E/D 冻结
    
    Z_s = Z[:, :32]   - 结构通道
    Z_d = Z[:, 32:]   - 细节通道
    """
    def __init__(self, latent_channels=128):
        super().__init__()
        self.struct_channels = config.STRUCT_CHANNELS  # 32
        self.detail_channels = config.DETAIL_CHANNELS  # 96
        
        self.encoder = Encoder(latent_channels=latent_channels)
        self.decoder = Decoder(latent_channels=latent_channels)
        
        # 辅助头 (Stage1 结构/细节解耦监督)
        self.struct_head = StructHead(struct_channels=self.struct_channels)
        self.detail_head = DetailHead(detail_channels=self.detail_channels)
        
        self.predictor = Predictor(
            embed_dim=latent_channels,
            num_layers=config.PRED_NUM_LAYERS,
            num_heads=config.PRED_NUM_HEADS,
            dim_ffn=config.PRED_DIM_FFN,
            dropout=config.PRED_DROPOUT,
            grid_size=config.LATENT_SPATIAL,
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def predict(self, z, mask):
        return self.predictor(z, mask)

    def split_latent(self, z):
        """将 latent 分离为结构/细节通道"""
        z_s = z[:, :self.struct_channels]   # [B, 32, H, W]
        z_d = z[:, self.struct_channels:]   # [B, 96, H, W]
        return z_s, z_d

    def forward_stage1(self, x):
        """
        阶段一前向: AE 重建 + 结构/细节辅助输出
        返回: x_recon, z, x_struct, x_detail
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        
        # 结构/细节辅助头
        z_s, z_d = self.split_latent(z)
        x_struct = self.struct_head(z_s)   # Z_s -> X_low 重建
        x_detail = self.detail_head(z_d)   # Z_d -> 细节残差重建
        
        return x_recon, z, x_struct, x_detail

    def forward_stage2(self, x, mask):
        """
        阶段二前向: masked latent completion
        x: [B, 3, 256, 256]
        mask: [B, N] bool
        返回: z (原始 latent), z_hat (补全后的 latent)
        """
        with torch.no_grad():
            z = self.encode(x)
        z_hat = self.predict(z, mask)
        return z, z_hat

    def freeze_ae(self):
        """冻结 Encoder 和 Decoder (含辅助头)"""
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False
        for p in self.struct_head.parameters():
            p.requires_grad = False
        for p in self.detail_head.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.decoder.eval()
        self.struct_head.eval()
        self.detail_head.eval()

    def unfreeze_ae(self):
        """解冻 Encoder 和 Decoder (含辅助头)"""
        for p in self.encoder.parameters():
            p.requires_grad = True
        for p in self.decoder.parameters():
            p.requires_grad = True
        for p in self.struct_head.parameters():
            p.requires_grad = True
        for p in self.detail_head.parameters():
            p.requires_grad = True


if __name__ == "__main__":
    model = LatentMAE(latent_channels=config.LATENT_CHANNELS)
    x = torch.randn(2, 3, 256, 256)

    # 阶段一测试
    x_recon, z, x_struct, x_detail = model.forward_stage1(x)
    z_s, z_d = model.split_latent(z)
    print(f"[阶段一] 输入: {x.shape}, Latent: {z.shape}, 重建: {x_recon.shape}")
    print(f"  Z_s: {z_s.shape}, Z_d: {z_d.shape}")
    print(f"  X_struct: {x_struct.shape}, X_detail: {x_detail.shape}")

    # 阶段二测试
    B = x.shape[0]
    masks = torch.stack([
        sample_mask(config.LATENT_SPATIAL, 0.5, "random")[0] for _ in range(B)
    ])
    z_orig, z_hat = model.forward_stage2(x, masks)
    print(f"[阶段二] Z: {z_orig.shape}, Z_hat: {z_hat.shape}")
    print(f"  mask 形状: {masks.shape}, 被 mask 比例: {masks.float().mean():.2f}")

    # 参数统计
    ae_params = sum(p.numel() for p in model.encoder.parameters()) + \
                sum(p.numel() for p in model.decoder.parameters())
    pred_params = sum(p.numel() for p in model.predictor.parameters())
    print(f"  AE 参数: {ae_params:,}")
    print(f"  Predictor 参数: {pred_params:,}")
    print(f"  总参数: {ae_params + pred_params:,}")
