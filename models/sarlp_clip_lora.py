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
# from peft import LoraConfig, get_peft_model, inject_adapter_in_model


__all__ = ['sarlp_clip_lora']


from clip_fine_tuning import clip
def get_clip_model(cfg):
    print("clip starts loading")
    try:
        # loading JIT archive
        clip_model = torch.jit.load(cfg.clip_model_path, map_location="cpu").eval()
        state_dict = None
        # clip_model = clip.build_model_conv_proj(state_dict,args).to(args.device)

    except RuntimeError:
        state_dict = torch.load(cfg.clip_model_path, map_location="cpu")
    if cfg.img_size > 224:
        clip_model = clip.build_model_conv_proj(state_dict or clip_model.state_dict(), cfg)
    if cfg.img_size == 224:
        clip_model = clip.build_model(state_dict or clip_model.state_dict())

    # clip_model = clip.build_model(state_dict or clip_model.state_dict())

    clip_model.float()  # 在服务器上跑时打开

    # 添加LoRA微调 - 仅对视觉编码器
    if cfg.use_lora:
        print("Applying LoRA to CLIP visual encoder")

        # 配置LoRA参数
        lora_config = LoraConfig(
            r=cfg.lora_r,  # LoRA秩 cfg.lora_r
            lora_alpha=cfg.lora_alpha,  # 缩放因子 cfg.lora_alpha
            target_modules=["q_proj", "k_proj", "v_proj", "c_proj",
                            # "layer4.0.conv1","layer4.0.conv2","layer4.0.conv3",
                            # "layer4.1.conv1","layer4.1.conv2","layer4.1.conv3",
                            # "layer4.2.conv1","layer4.2.conv2","layer4.2.conv3",
                            # "layer3.conv1", "layer3.conv2", "layer3.conv3"
                            # "layer4.conv1", "layer4.conv2", "layer4.conv3"

                            # "layer1.0.conv1", "layer1.0.conv2", "layer1.0.conv3",
                            # "layer1.1.conv1", "layer1.1.conv2", "layer1.1.conv3",
                            # "layer1.2.conv1", "layer1.2.conv2", "layer1.2.conv3",

                            "layer2.0.conv1", "layer2.0.conv2", "layer2.0.conv3",
                            "layer2.1.conv1", "layer2.1.conv2", "layer2.1.conv3",
                            "layer2.2.conv1", "layer2.2.conv2", "layer2.2.conv3",
                            "layer2.3.conv1", "layer2.3.conv2", "layer2.3.conv3",

                            "layer3.0.conv1","layer3.0.conv2","layer3.0.conv3",
                            "layer3.1.conv1","layer3.1.conv2","layer3.1.conv3",
                            "layer3.2.conv1", "layer3.2.conv2", "layer3.2.conv3",
                            "layer3.3.conv1", "layer3.3.conv2", "layer3.3.conv3",
                            "layer3.4.conv1", "layer3.4.conv2", "layer3.4.conv3",
                            "layer3.5.conv1", "layer3.5.conv2", "layer3.5.conv3",
                            "layer3.6.conv1", "layer3.6.conv2", "layer3.6.conv3",
                            "layer3.7.conv1", "layer3.7.conv2", "layer3.7.conv3",
                            "layer3.8.conv1", "layer3.8.conv2", "layer3.8.conv3",
                            "layer3.9.conv1", "layer3.9.conv2", "layer3.9.conv3",
                            "layer3.10.conv1", "layer3.10.conv2", "layer3.10.conv3",
                            "layer3.11.conv1", "layer3.11.conv2", "layer3.11.conv3",
                            "layer3.12.conv1", "layer3.12.conv2", "layer3.12.conv3",
                            "layer3.13.conv1", "layer3.13.conv2", "layer3.13.conv3",
                            "layer3.14.conv1", "layer3.14.conv2", "layer3.14.conv3",
                            "layer3.15.conv1", "layer3.15.conv2", "layer3.15.conv3",
                            "layer3.16.conv1", "layer3.16.conv2", "layer3.16.conv3",
                            "layer3.17.conv1", "layer3.17.conv2", "layer3.17.conv3",
                            "layer3.18.conv1", "layer3.18.conv2", "layer3.18.conv3",
                            "layer3.19.conv1", "layer3.19.conv2", "layer3.19.conv3",
                            "layer3.20.conv1", "layer3.20.conv2", "layer3.20.conv3",
                            "layer3.21.conv1", "layer3.21.conv2", "layer3.21.conv3",
                            "layer3.22.conv1", "layer3.22.conv2", "layer3.22.conv3",


                            "layer4.0.conv1","layer4.0.conv2","layer4.0.conv3",
                            "layer4.1.conv1","layer4.1.conv2","layer4.1.conv3",
                            "layer4.2.conv1","layer4.2.conv2","layer4.2.conv3",
                            ],  # 目标模块
            lora_dropout=0.01,  # Dropout率 cfg.lora_dropout
            bias="none",  # 偏置处理方式
            modules_to_save=[
                             # "layer4.0.bn1", "layer4.0.bn2", "layer4.0.bn3",
                             # "layer4.1.bn1", "layer4.1.bn2", "layer4.1.bn3",
                             "layer4.2.bn1", "layer4.2.bn2", "layer4.2.bn3",
                             ],  # 额外训练的参数
            # target_modules=[
            # "q_proj",
            # "k_proj",
            # "v_proj",
            # "c_proj",
            # # "attnpool.q_proj",
            # # "attnpool.k_proj",
            # # "attnpool.v_proj",
            # # "attnpool.c_proj",
            # "layer4.2.conv1",
            # "layer4.2.conv2",
            # "layer4.2.conv3",
            # # "proj"
            # ],
            # modules_to_save=[
            #     "attnpool",  # 整个注意力池化模块
            #     "bn1",
            #     "layer4.2.bn1",
            #     "layer4.2.bn2",
            #     "layer4.2.bn3",
            #     # "proj"  # 确保投影头完全训练
            # ],
        )

        # 仅对视觉编码器应用LoRA
        visual_model = clip_model.visual
        visual_model = get_peft_model(visual_model, lora_config)
        visual_model.print_trainable_parameters()

        # 替换原始视觉编码器
        clip_model.visual = visual_model

    print("clip loading complete")
    return clip_model



class LowRankBilinearAttention(nn.Module):
    """
    Low-rank bilinear attention network.
    """

    def __init__(self, dim1, dim2, att_dim=2048):
        """
        :param dim1: feature size of encoded images
        :param dim2: feature size of encoded labels
        :param att_dim: size of the attention network
        """
        super().__init__()
        self.linear1 = nn.Linear(dim1, att_dim, bias=False)  # linear layer to transform encoded image
        self.linear2 = nn.Linear(dim2, att_dim, bias=False)  # linear layer to transform decoder's output
        self.hidden_linear = nn.Linear(att_dim, att_dim)  # linear layer to calculate values to be softmax-ed
        self.target_linear = nn.Linear(att_dim, 1)
        self.tanh = nn.Tanh()
        # self.softmax = nn.Softmax(dim=-1)  # softmax layer to calculate weights

    def forward(self, x1, x2, tau=1.0):
        """
        Forward propagation.
        :param
            x1: a tensor of dimension (B, num_pixels, dim1)
            x2: a tensor of dimension (B, num_labels, dim2)
        """
        _x1 = self.linear1(x1).unsqueeze(dim=1)  # (B, 1, num_pixels, att_dim)
        _x2 = self.linear2(x2).unsqueeze(dim=2)  # (B, num_labels, 1, att_dim)
        t = self.hidden_linear(self.tanh(_x1 * _x2))
        alpha = self.target_linear(t).squeeze(-1)  # B, num_labels, num_pixels  [B, C, P]
        # alpha = self.softmax(temp / tau)  # (B, num_labels, num_pixels)
        # alpha = self.tanh(temp / tau)  # (B, num_labels, num_pixels)
        return alpha


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



class SARLP_LoRA(nn.Module):
    def __init__(self, backbone, feat_dim, cfg):
        super().__init__()
        self.cfg = cfg
        # self.backbone = backbone   # rn101

        self.backbone = backbone.visual  # rn101_clip
        # 冻结非LoRA参数（如果启用LoRA）
        if cfg.use_lora:
            self.freeze_non_lora_parameters()
        self.dtype = backbone.dtype  # rn101_clip

        self.position_embedding = build_position_encoding(feat_dim, cfg.arch, 'sine', cfg.img_size)
        self.encoder = TransformerEncoder(feat_dim, cfg.num_heads, cfg.num_layers, cfg.normalize_before)

        if self.cfg.embed_type == 'random':
            self.embeddings = nn.Parameter(torch.empty((cfg.num_classes, 768)))
            nn.init.kaiming_uniform_(self.embeddings, a=math.sqrt(5))
        else:
            self.embeddings = torch.from_numpy(np.load(cfg.embed_path)).float().cuda()
        text_dim = self.embeddings.shape[-1]
        # self.attention = LowRankBilinearAttention(feat_dim, text_dim, 1024)  # pre_COCO ex
        # self.attention2 = LowRankBilinearAttention(feat_dim, feat_dim, 256)
        self.attention2 = LowRankBilinearAttention(2048, 2048, 1024)

        self.mlp = nn.Sequential(
            nn.Conv1d(feat_dim, 2048, 1),
            nn.BatchNorm1d(2048),
            # nn.GELU(),
            # # # nn.ReLU(inplace=True),
            # nn.Conv1d(512, 512, 1),
        )
        self.fc0 = nn.Conv1d(2048, cfg.num_classes, 1)
        # self.fc0_global = nn.Linear(2048, cfg.num_classes)
        self.fc2 = nn.Sequential(
            nn.Linear(2048+2048, 2048+2048),
            # nn.Linear(2048, 2048),
            nn.GELU(),
            # nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048+2048, cfg.num_classes),
            # nn.Linear(2048, cfg.num_classes),
            # nn.Linear(cfg.num_classes, cfg.num_classes),  # CC_1
            # nn.Linear(cfg.num_classes, cfg.num_classes),  # CC_2
        )
        self.criterion = get_loss_fn(cfg)
        self.gsp = nn.AdaptiveAvgPool1d(1)
        self.ttd = nn.Sequential(
            nn.Conv1d(text_dim+512, text_dim+512, 1),
            nn.BatchNorm1d(text_dim+512),
            nn.GELU(),
            # nn.ReLU(inplace=True),
            nn.Conv1d(text_dim+512, 2048, 1),
        )

    def freeze_non_lora_parameters(self):
        """冻结所有非LoRA参数"""
        for name, param in self.backbone.named_parameters():
            # 只训练LoRA参数和指定保留的参数
            if not any(key in name for key in ["lora_", "class_embedding", "positional_embedding"]):
                param.requires_grad = False
        print("Frozen non-LoRA parameters")

    # def save_lora_weights(self, path):
    #     """仅保存LoRA适配器权重"""
    #     if self.cfg.use_lora:
    #         self.backbone.save_pretrained(path)
    #         print(f"Saved LoRA weights to {path}")
    #     else:
    #         print("LoRA not enabled, skipping weight save")

    def clip_448_encoder(self, x):
        """
        x: [B, 3, 448, 448]
        返回: [B, 1 + 14*14, 512]
        """
        B = x.size(0)

        # visualize_crops_and_original(x, num_images=1)

        # 1) 切成 4 张 [B,3,224,224] -> [B,4,3,224,224]
        crops = torch.stack([
            x[:, :,     :224,     :224],  # top-left
            x[:, :,     :224, 224:   ],   # top-right
            x[:, :, 224:   ,     :224],   # bottom-left
            x[:, :, 224:   , 224:   ],    # bottom-right
        ], dim=1)

        # 2) 拼 batch -> [B*4,3,224,224]
        crops_flat = crops.view(B * 4, 3, 224, 224).type(self.dtype)

        # 3) 编码 -> [B*4, 1+P, D]
        feat = self.backbone(crops_flat)
        _, S, D = feat.shape  # S = 1 + P
        P = S - 1

        # 4) reshape 回 [B,4,1+P,D]
        feat = feat.view(B, 4, S, D)

        # 5) 平均 CLS -> [B,1,D]
        cls_tokens = feat[:, :, 0, :]                  # [B,4,D]
        # new_cls     = cls_tokens.mean(dim=1, keepdim=True)  # [B,1,D]
        new_cls = cls_tokens.sum(dim=1, keepdim=True)  # [B,1,D]

        # 6) 对每个 crop 的 patch tokens 先 reshape 成网格 -> [B,4,grid,grid,D]
        grid = int(math.sqrt(P))  # e.g. 7
        assert grid * grid == P, "patch 数量不是完全平方"
        patch_grid = feat[:, :, 1:, :].view(B, 4, grid, grid, D)

        # 7) 按照裁剪位置拼回 14×14 大网格
        #    crops 顺序是 [TL, TR, BL, BR]
        top    = torch.cat([patch_grid[:,0], patch_grid[:,1]], dim=2)  # [B, 7, 14, D]
        bottom = torch.cat([patch_grid[:,2], patch_grid[:,3]], dim=2)  # [B, 7, 14, D]
        full   = torch.cat([top, bottom], dim=1)                       # [B,14,14,D]

        # 8) flatten -> [B,14*14,D]
        flat_patches = full.view(B, grid*2 * grid*2, D)  # 14*14

        # 9) 拼回 CLS -> [B, 1+196, D]
        out = torch.cat([new_cls, flat_patches], dim=1)

        return out


    def forward(self, x, y=None, mask_label=None):
        # x = self.backbone(x)  # [B, P, 2048]   resnet101:7x7, 2048  # image_pre
        # input x : [B, 3, 448, 448]
        if self.cfg.img_size > 224:
            x, _ = self.backbone(x.type(self.dtype))   # clip_pre  [B, 512, 1+Patch]  default
            x = x.transpose(1, 2)  # [B, 1+Patch, 512]
            # x = self.clip_448_encoder(x)  # [B, 1+ P, 512]  448裁剪
            # x = self.backbone(x.type(self.dtype))  # clip_pre  [B, 512, 1+Patch]  448-vit
        if self.cfg.img_size == 224:
            x = self.backbone(x.type(self.dtype))   # clip_pre  [B, 512, 1+Patch]
        # x = x.detach()

        # normalized features
        # x_g = x[:, 0]  # [B, 1, 512]
        x = x / x.norm(dim=-1, keepdim=True)  # [B, 1+p, d]
        x_g = x[:, 0]  # [B, 1, 512]
        # x_label = x_label / x_label.norm(dim=1, keepdim=True)  # [C, d]

        x_l = x[:, 1:]  # [B, Patch, 512]
        # if self.cfg.rm_cls and 'vit' in self.cfg.arch:
        #     x = x[:, 1:]
        # x = x.transpose(1, 2)  # [B, 1+Patch, 512]
        # x = self.backbone(x.type(self.dtype))  # clip_pre  [B, 512, 1+Patch] vitb32
        pos = None
        if self.cfg.pos:
            pos = self.position_embedding(x_l)
            pos = torch.flatten(pos, 2).transpose(1, 2)
        # x = self.mlp(x.transpose(1, 2)).transpose(1, 2)
        x_l = self.encoder(x_l, pos=pos)  # [B, P, dv]
        x_l = self.mlp(x_l.transpose(1, 2)).transpose(1, 2)
        # x = self.mlp(x.transpose(1, 2)).transpose(1, 2) + x

        # x_l = x[:, 1:]  # [B, Patch, 512]
        # x_g = x[:, 0]  # [B, 1, 512]
        # x_g = self.gsp(x_l.transpose(1, 2)).transpose(1, 2)

        # SRFL
        # x_g = self.gsp(x_l.transpose(1, 2)).transpose(1, 2)
        v = self.embeddings.unsqueeze(0).repeat(x_l.shape[0], 1, 1)  # [B, C, dt]  标签嵌入
        # v_norm = v / v.norm(dim=-1, keepdim=True)
        # vx = torch.cat((v, x_g.repeat(1, v.shape[1], 1)), dim=-1)  # [B, C, dt+dv]
        vx = torch.cat((v, x_g.unsqueeze(1).repeat(1, v.shape[1], 1)), dim=-1)  # [B, C, dt+dv]
        v = self.ttd(vx.transpose(1, 2)).transpose(1, 2)  # dt --> 1024

        # sam
        alpha0 = self.fc0(x_l.transpose(1, 2))  # [B, class, P]
        logits0 = alpha0.topk(1, dim=-1)[0].mean(dim=-1)
        # logits0 = self.fc0_global(x_g)
        # logits0 = torch.mul(torch.softmax(alpha0.transpose(1, 2), dim=1), alpha0.transpose(1, 2)).sum(dim=1)

        # SGFE
        # alpha2 = self.OT_attention(x, v, 1).transpose(1, 2)  # [B, P, C]  # c
        alpha2 = self.attention2(x_l, v, 1).transpose(1, 2)  # [B, P, C]  #
        f = torch.matmul(torch.softmax(alpha2, dim=-1), v)  # [B, P, 1024]  # softmax
        # x_g.unsqueeze(1).repeat(1, v.shape[1], 1)
        f = torch.cat((f, x_l), dim=-1)  # [B, P, 1024+1024]
        # f = torch.cat((f, x_g.unsqueeze(1).repeat(1, f.shape[1], 1)), dim=-1)  # [B, P, 1024+1024]
        # f = torch.cat((f, x_l), dim=-1)  # [B, P, 1024+1024]

        alpha3 = self.fc2(f)  # [B, P, C]
        logits1 = torch.mul(torch.softmax(alpha3, dim=1), alpha3).sum(dim=1)
        # logits2 = logits2 @ A

        # abs_A = torch.abs(A)
        # sum_abs_cols = abs_A.sum(dim=0)
        # l1_loss = sum_abs_cols.max()
        # # adj_numpy = adj.cpu().detach().numpy()

        ce_loss = 0
        # l1_loss = 0
        bce0 = 0
        bce1 = 0
        # OT_loss = 0
        # last_y_bar = y_bar
        # y_sam = 0
        # y_fc = 0
        L_norm = 0
        L_clip = 0
        if self.training:

            # 伪标签
            # _pre = torch.mul(torch.softmax(alpha2, dim=1), alpha2).sum(dim=1)  # [B, C]
            # pre = torch.sigmoid(_pre)
            # mask = mask_label.reshape(pre.shape[0], -1)  # [B, C]  ，[0, 1],1表示可见，0表示不可见
            # y_new = (1 - mask) * pre + mask * y.squeeze(1)
            if self.cfg.partial == 1:
                bce0 = self.criterion(logits0, y)
                bce1 = self.criterion(logits1, y)
            else:
                # pre0 = torch.sigmoid(logits0 * logits0.max(dim=1, keepdim=True).values)
                # pre1 = torch.sigmoid(logits1 * logits1.max(dim=1, keepdim=True).values)
                # pre1 = torch.sigmoid(logits1 * (logits1.max(dim=1)).unsqueeze(1).repeat(logits1.size(0), logits1.size(1)))
                pre0 = torch.sigmoid(logits0)
                # pre1 = torch.sigmoid(logits1)
                pre1 = torch.sigmoid(logits1 * logits1.max(dim=1, keepdim=True).values)

                mask = mask_label.reshape(logits1.shape[0], -1)  # [B, C]  ，[0, 1],1表示可见，0表示不可见
                y_sam = (1-mask)*pre1 + mask*y.squeeze(1)
                y_fc = (1-mask)*pre0 + mask*y.squeeze(1)
                # L_norm = torch.norm(y_sam - y_fc)**2
                # clip
                # L_clip = (self.criterion(y_sam, y_fc.detach()) + self.criterion(y_fc, y_sam.detach()))/2
                # 阈值裁剪
                # clamp = 0.05
                # y_sam[y_sam < clamp] = 0
                # y_sam[y_sam > (1-clamp)] = 1
                # y_fc[y_fc < clamp] = 0
                # y_fc[y_fc > (1 - clamp)] = 1
                # y_hat = (y_sam + y_fc)/2

                bce0 = self.criterion(logits0, y_sam.detach())
                # bce1 = self.criterion(logits1, y_fc.detach())
                bce1 = self.criterion(logits1, y.squeeze(1), MASK=True)
                # bce0 = self.criterion(logits0, y_hat.detach())
                # bce1 = self.criterion(logits1, y_hat.detach())
            # ce_loss = 1 * bce1 + 4 * bce0 + 0 * L_norm + 0 * L_clip
            # ce_loss = bce2 + self.cfg.lamda * (bce1 + bce0)

            ce_loss = 1 * bce1 + 0.1 * bce0 + 0 * L_norm + 0 * L_clip

            # 逐步更新伪标记
            # y_sam = ((1 - mask) * pre0 + mask * y.squeeze(1)).unsqueeze(1)
            # y_fc = ((1 - mask) * pre1 + mask * y.squeeze(1)).unsqueeze(1)

        return {
            'logits': torch.sigmoid(logits1),
            'alpha': alpha3.transpose(1, 2),  # alpha2.transpose(1, 2),
            'ce_loss': ce_loss,
            'loss_sam': bce0,
            'loss_fc': bce1,
            # 'y_sam': y_sam,
            # 'y_fc': y_fc,
        }


@register_model
def sarlp_clip_lora(cfg):
    # default
    # backbone, feat_dim = create_backbone(cfg.arch, img_size=cfg.img_size)
    # model = SARLP(backbone, feat_dim, cfg)

    # clip
    backbone = get_clip_model(cfg)
    model = SARLP_LoRA(backbone, feat_dim=512, cfg=cfg)
    print(model)
    return model