import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import config


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    if embed_dim % 4 != 0:
        raise ValueError("embed_dim 必须能被 4 整除")
    y, x = torch.meshgrid(
        torch.arange(grid_size, dtype=torch.float32),
        torch.arange(grid_size, dtype=torch.float32),
        indexing="ij",
    )
    omega = torch.arange(embed_dim // 4, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim // 4)))
    out_y = y.reshape(-1, 1) * omega.reshape(1, -1)
    out_x = x.reshape(-1, 1) * omega.reshape(1, -1)
    pos = torch.cat([out_y.sin(), out_y.cos(), out_x.sin(), out_x.cos()], dim=1)
    return pos


def ratio_to_num_mask(num_tokens, mask_ratio):
    num_mask = int(round(num_tokens * float(mask_ratio)))
    return max(1, min(num_mask, num_tokens - 1))


def random_mask(grid_size, mask_ratio):
    num_tokens = grid_size * grid_size
    num_mask = ratio_to_num_mask(num_tokens, mask_ratio)
    perm = torch.randperm(num_tokens)
    mask = torch.zeros(num_tokens, dtype=torch.bool)
    mask[perm[:num_mask]] = True
    return mask, mask.float().mean().item()


def block_mask(grid_size, mask_ratio):
    num_tokens = grid_size * grid_size
    target = ratio_to_num_mask(num_tokens, mask_ratio)
    best = None
    best_diff = 10 ** 9
    for h in range(1, grid_size + 1):
        for w in range(1, grid_size + 1):
            aspect = h / max(w, 1)
            if aspect < 0.5 or aspect > 2.0:
                continue
            area = h * w
            if area >= num_tokens:
                continue
            diff = abs(area - target)
            if diff < best_diff:
                best_diff = diff
                best = (h, w)
    if best is None:
        return random_mask(grid_size, mask_ratio)
    h, w = best
    top = random.randint(0, grid_size - h)
    left = random.randint(0, grid_size - w)
    grid = torch.zeros(grid_size, grid_size, dtype=torch.bool)
    grid[top:top + h, left:left + w] = True
    mask = grid.reshape(-1)
    return mask, mask.float().mean().item()


def stripe_mask(grid_size, mask_ratio):
    num_mask = ratio_to_num_mask(grid_size, mask_ratio)
    horizontal = random.random() < 0.5
    grid = torch.zeros(grid_size, grid_size, dtype=torch.bool)
    start = random.randint(0, grid_size - num_mask)
    if horizontal:
        grid[start:start + num_mask, :] = True
    else:
        grid[:, start:start + num_mask] = True
    mask = grid.reshape(-1)
    return mask, mask.float().mean().item()


def half_mask(grid_size, mask_ratio=None):
    horizontal = random.random() < 0.5
    grid = torch.zeros(grid_size, grid_size, dtype=torch.bool)
    if horizontal:
        if random.random() < 0.5:
            grid[: grid_size // 2, :] = True
        else:
            grid[grid_size // 2 :, :] = True
    else:
        if random.random() < 0.5:
            grid[:, : grid_size // 2] = True
        else:
            grid[:, grid_size // 2 :] = True
    mask = grid.reshape(-1)
    return mask, mask.float().mean().item()


def sample_mask(grid_size, mask_ratio, mask_type="random"):
    if mask_type == "random":
        return random_mask(grid_size, mask_ratio)
    if mask_type == "block":
        return block_mask(grid_size, mask_ratio)
    if mask_type == "stripe":
        return stripe_mask(grid_size, mask_ratio)
    if mask_type == "half":
        return half_mask(grid_size, mask_ratio)
    raise ValueError(f"未知 mask_type: {mask_type}")


def sample_train_mask(grid_size):
    if random.random() < config.TRAIN_MASK_MAIN_PROB:
        ratio = random.choice(config.TRAIN_MASK_RATIOS_MAIN)
    else:
        ratio = random.choice(config.TRAIN_MASK_RATIOS_AUX)
    mask_type = random.choices(
        config.TRAIN_MASK_TYPES,
        weights=config.TRAIN_MASK_TYPE_PROBS,
        k=1,
    )[0]
    mask, actual_ratio = sample_mask(grid_size, ratio, mask_type)
    return mask, actual_ratio, mask_type, ratio


def batch_sample_masks(batch_size, grid_size, mask_ratio, mask_type, device):
    if mask_type == "random":
        num_tokens = grid_size * grid_size
        num_mask = ratio_to_num_mask(num_tokens, mask_ratio)
        noise = torch.rand(batch_size, num_tokens, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        masks = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)
        masks.scatter_(1, ids_shuffle[:, :num_mask], True)
        actual_ratios = [float(num_mask) / num_tokens] * batch_size
        return masks, actual_ratios
    masks = []
    actual_ratios = []
    for _ in range(batch_size):
        mask, actual_ratio = sample_mask(grid_size, mask_ratio, mask_type)
        masks.append(mask)
        actual_ratios.append(actual_ratio)
    return torch.stack(masks, dim=0).to(device), actual_ratios


def batch_sample_train_masks(batch_size, grid_size, device):
    num_tokens = grid_size * grid_size
    main_count = int(batch_size * config.TRAIN_MASK_MAIN_PROB)
    aux_count = batch_size - main_count
    ratios = []
    if main_count > 0:
        ratios.extend(random.choices(config.TRAIN_MASK_RATIOS_MAIN, k=main_count))
    if aux_count > 0:
        ratios.extend(random.choices(config.TRAIN_MASK_RATIOS_AUX, k=aux_count))
    random.shuffle(ratios)
    type_weights = config.TRAIN_MASK_TYPE_PROBS
    type_choices = random.choices(config.TRAIN_MASK_TYPES, weights=type_weights, k=batch_size)
    random_indices = [i for i, t in enumerate(type_choices) if t == "random"]
    other_indices = [i for i, t in enumerate(type_choices) if t != "random"]
    masks = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)
    meta = [None] * batch_size
    if random_indices:
        for idx in random_indices:
            num_mask = ratio_to_num_mask(num_tokens, ratios[idx])
            noise = torch.rand(num_tokens, device=device)
            ids = torch.argsort(noise)
            masks[idx, ids[:num_mask]] = True
            actual = float(num_mask) / num_tokens
            meta[idx] = {"actual_ratio": actual, "mask_type": "random", "target_ratio": ratios[idx]}
    for idx in other_indices:
        mask, actual_ratio = sample_mask(grid_size, ratios[idx], type_choices[idx])
        masks[idx] = mask.to(device)
        meta[idx] = {"actual_ratio": actual_ratio, "mask_type": type_choices[idx], "target_ratio": ratios[idx]}
    return masks, meta


def patch_mask_to_pixel_mask(mask, img_size=None, patch_size=None):
    if img_size is None:
        img_size = config.IMG_SIZE
    if patch_size is None:
        patch_size = config.PATCH_SIZE
    squeeze = False
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
        squeeze = True
    grid_size = img_size // patch_size
    pixel_mask = mask.reshape(mask.shape[0], grid_size, grid_size)
    pixel_mask = pixel_mask.unsqueeze(1).float()
    pixel_mask = pixel_mask.repeat_interleave(patch_size, dim=2)
    pixel_mask = pixel_mask.repeat_interleave(patch_size, dim=3)
    if squeeze:
        pixel_mask = pixel_mask[0]
    return pixel_mask


def erode_pixel_mask(pixel_mask, erosion_px):
    if erosion_px <= 0:
        return pixel_mask
    kernel = 2 * erosion_px + 1
    inv = 1.0 - pixel_mask.float()
    dilated_inv = F.max_pool2d(inv, kernel_size=kernel, stride=1, padding=erosion_px)
    eroded = 1.0 - dilated_inv
    return (eroded > 0.5).float()


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x, key_padding_mask=None):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerStack(nn.Module):
    def __init__(self, depth, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, key_padding_mask=None):
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class PixelMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_size = config.IMG_SIZE
        self.patch_size = config.PATCH_SIZE
        self.grid_size = config.GRID_SIZE
        self.num_tokens = config.NUM_TOKENS
        self.patch_dim = config.PATCH_DIM
        self.enc_embed_dim = config.ENC_EMBED_DIM
        self.dec_embed_dim = config.DEC_EMBED_DIM

        self.patch_embed = nn.Conv2d(
            config.IN_CHANNELS,
            self.enc_embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        enc_pos = get_2d_sincos_pos_embed(self.enc_embed_dim, self.grid_size)
        dec_pos = get_2d_sincos_pos_embed(self.dec_embed_dim, self.grid_size)
        self.register_buffer("enc_pos_embed", enc_pos.unsqueeze(0), persistent=False)
        self.register_buffer("dec_pos_embed", dec_pos.unsqueeze(0), persistent=False)

        self.encoder = TransformerStack(
            depth=config.ENC_DEPTH,
            dim=self.enc_embed_dim,
            num_heads=config.ENC_NUM_HEADS,
            mlp_ratio=config.ENC_MLP_RATIO,
            dropout=config.ENC_DROPOUT,
        )
        self.decoder_embed = nn.Linear(self.enc_embed_dim, self.dec_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.dec_embed_dim))
        self.decoder = TransformerStack(
            depth=config.DEC_DEPTH,
            dim=self.dec_embed_dim,
            num_heads=config.DEC_NUM_HEADS,
            mlp_ratio=config.DEC_MLP_RATIO,
            dropout=config.DEC_DROPOUT,
        )
        self.decoder_pred = nn.Linear(self.dec_embed_dim, self.patch_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.mask_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def patchify(self, x):
        p = self.patch_size
        b, c, h, w = x.shape
        if h != self.img_size or w != self.img_size:
            raise ValueError(f"输入尺寸必须为 {self.img_size}x{self.img_size}")
        x = x.reshape(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(b, self.num_tokens, p * p * c)
        return x

    def unpatchify(self, patches):
        p = self.patch_size
        b = patches.shape[0]
        x = patches.reshape(b, self.grid_size, self.grid_size, p, p, config.IN_CHANNELS)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(b, config.IN_CHANNELS, self.img_size, self.img_size)
        return x

    def embed_patches(self, x):
        tokens = self.patch_embed(x)
        tokens = tokens.flatten(2).transpose(1, 2)
        return tokens

    def encode_visible(self, tokens, mask):
        bsz, num_tokens, dim = tokens.shape
        pos = self.enc_pos_embed.expand(bsz, -1, -1).to(tokens.dtype)
        tokens_with_pos = tokens + pos
        visible_mask = ~mask
        lengths = visible_mask.sum(dim=1)
        max_len = int(lengths.max().item())
        batch_idx = torch.arange(bsz, device=tokens.device).unsqueeze(1).expand(-1, num_tokens)
        token_idx = torch.arange(num_tokens, device=tokens.device).unsqueeze(0).expand(bsz, -1)
        visible_pos = torch.where(visible_mask, token_idx, torch.tensor(num_tokens, device=tokens.device))
        sorted_pos, sort_idx = visible_pos.sort(dim=1)
        gathered = tokens_with_pos.gather(1, sort_idx.unsqueeze(-1).expand(-1, -1, dim))
        padded = gathered[:, :max_len, :]
        seq_idx = torch.arange(max_len, device=tokens.device).unsqueeze(0).expand(bsz, -1)
        key_padding_mask = seq_idx >= lengths.unsqueeze(1)
        encoded = self.encoder(padded, key_padding_mask=key_padding_mask)
        return encoded, lengths.tolist()

    def decode_full(self, encoded_visible, lengths, mask):
        bsz = mask.shape[0]
        num_tokens = self.num_tokens
        dec_dim = self.dec_embed_dim
        device = mask.device
        visible_mask = ~mask
        dec_tokens = self.mask_token.expand(bsz, num_tokens, -1).clone()
        lengths_t = torch.tensor(lengths, device=device)
        max_len = int(lengths_t.max().item())
        seq_idx = torch.arange(max_len, device=device).unsqueeze(0).expand(bsz, -1)
        valid_mask = seq_idx < lengths_t.unsqueeze(1)
        encoded_valid = encoded_visible[:, :max_len, :]
        encoded_proj = self.decoder_embed(encoded_valid)
        token_idx = torch.arange(num_tokens, device=device).unsqueeze(0).expand(bsz, -1)
        visible_pos = torch.where(visible_mask, token_idx, torch.tensor(num_tokens, device=device))
        sorted_pos, sort_idx = visible_pos.sort(dim=1)
        inv_idx = torch.zeros_like(sort_idx)
        inv_idx.scatter_(1, sort_idx, torch.arange(num_tokens, device=device).unsqueeze(0).expand(bsz, -1))
        visible_inv = inv_idx[visible_mask]
        batch_indices = torch.arange(bsz, device=device).unsqueeze(1).expand_as(visible_mask)[visible_mask]
        token_indices = token_idx[visible_mask]
        enc_indices = visible_inv
        enc_indices = enc_indices.clamp(max=max_len - 1)
        dec_tokens[batch_indices, token_indices] = encoded_proj[batch_indices, enc_indices]
        dec_tokens = dec_tokens + self.dec_pos_embed.to(dec_tokens.dtype)
        dec_tokens = self.decoder(dec_tokens)
        pred_patches = self.decoder_pred(dec_tokens)
        return pred_patches

    def forward(self, x, mask):
        if mask.dtype != torch.bool:
            mask = mask.bool()
        tokens = self.embed_patches(x)
        encoded_visible, lengths = self.encode_visible(tokens, mask)
        pred_patches = self.decode_full(encoded_visible, lengths, mask)
        return pred_patches

    def reconstruct(self, x, mask, copy_back=True):
        gt_patches = self.patchify(x)
        pred_patches = self.forward(x, mask)
        if copy_back:
            recon_patches = gt_patches.clone()
            recon_patches[mask] = pred_patches[mask]
        else:
            recon_patches = pred_patches
        recon = self.unpatchify(recon_patches).clamp(0.0, 1.0)
        pred_full = self.unpatchify(pred_patches).clamp(0.0, 1.0)
        return recon, pred_full, pred_patches, gt_patches


if __name__ == "__main__":
    model = PixelMAE()
    x = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE)
    masks, _ = batch_sample_masks(2, config.GRID_SIZE, 0.6, "random", x.device)
    pred = model(x, masks)
    recon, pred_full, _, _ = model.reconstruct(x, masks, copy_back=True)
    print(pred.shape, recon.shape, pred_full.shape)
