# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: Silence Xie
# Created on: 2025-5-9
# Email: silencexie@163.com
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import math
import copy
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import torch.nn.functional as F

from .utils import Element_Wise_Layer, gen_A, gen_adj, gen_adj_L
from .factory import register_model, create_backbone
from lib.util import get_loss_fn



__all__ = ['baseline']


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, maxH=30, maxW=30):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.maxH = maxH
        self.maxW = maxW
        pe = self._gen_pos_buffer()
        self.register_buffer('pe', pe)

    def _gen_pos_buffer(self):
        _eyes = torch.ones((1, self.maxH, self.maxW))
        y_embed = _eyes.cumsum(1, dtype=torch.float32)
        x_embed = _eyes.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def forward(self, input: Tensor):
        x = input
        return self.pe.repeat((x.size(0), 1, 1, 1))


def build_position_encoding(hidden_dim, arch, position_embedding, img_size):
    N_steps = hidden_dim // 2

    if arch in ['CvT_w24'] or 'vit' in arch:
        downsample_ratio = 16
    else:
        downsample_ratio = 32

    if position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        assert img_size % 32 == 0, "args.img_size ({}) % 32 != 0".format(img_size)
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True, maxH=img_size // downsample_ratio,
                                                   maxW=img_size // downsample_ratio)
    else:
        raise ValueError(f"not supported {position_embedding}")

    return position_embedding


class TransformerEncoder(nn.Module):

    def __init__(self, d_model, nhead, num_layers, normalize_before, norm=None):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(TransformerEncoderLayer(d_model, nhead, normalize_before=normalize_before)) for _ in
             range(num_layers)])
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2, corr = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]

        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)



class BASE(nn.Module):
    def __init__(self, backbone, feat_dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone   # rn101

        # self.backbone = backbone.visual  # rn101_clip
        # self.dtype = backbone.dtype  # rn101_clip

        # self.position_embedding = build_position_encoding(feat_dim, cfg.arch, 'sine', cfg.img_size)
        # self.encoder = TransformerEncoder(feat_dim, cfg.num_heads, cfg.num_layers, cfg.normalize_before)

        self.fc0 = nn.Conv1d(feat_dim, cfg.num_classes, 1)
        self.criterion = get_loss_fn(cfg)

    def forward(self, x, y=None, mask_label=None):
        x = self.backbone(x)  # [B, P, 2048]   resnet101:7x7, 2048  # image_pre
        # x, _ = self.backbone(x.type(self.dtype))   # clip_pre
        # if self.cfg.rm_cls and 'vit' in self.cfg.arch:
        #     x = x[:, 1:]
        # pos = None
        # if self.cfg.pos:
        #     pos = self.position_embedding(x)
        #     pos = torch.flatten(pos, 2).transpose(1, 2)
        # _x = self.encoder(x, pos=pos)  # [B, P, dv]

        # alpha0 = self.fc0(x.transpose(1, 2))  # [B, class, P]
        # logits0 = alpha0.topk(1, dim=-1)[0].mean(dim=-1)

        alpha0 = self.fc0(x.transpose(1, 2)).transpose(1, 2)  # [B, P, C]
        logits0 = torch.mul(torch.softmax(alpha0, dim=1), alpha0).sum(dim=1)

        ce_loss = 0
        bce0 = 0
        bce1 = 0

        if self.training:
            if self.cfg.partial == 1:
                bce0 = self.criterion(logits0, y)
            else:
                # 缺失标记不参与损失计算，且忽略标记全为unseen的样本
                bce0 = self.criterion(logits0, y.squeeze(1), MASK=True)
                # # 缺失标记设置成负标记
                # y[y == -1] = 0
                # bce0 = self.criterion(logits0, y.squeeze(1))
            ce_loss = bce0
        return {
            'logits': torch.sigmoid(logits0),
            'alpha': alpha0,  # alpha2.transpose(1, 2),
            'ce_loss': ce_loss,
            'loss_sam': bce0,
            'loss_fc': bce1,
            # 'y_sam': y_sam,
            # 'y_fc': y_fc,
        }


@register_model
def baseline(cfg):
    # default
    backbone, feat_dim = create_backbone(cfg.arch, img_size=cfg.img_size)
    model = BASE(backbone, feat_dim, cfg)

    # clip
    # backbone = get_clip_model(cfg)
    # model = SARLP(backbone, feat_dim=512, cfg=cfg)
    return model