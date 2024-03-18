import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from base_network import LayerNorm


class DGFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(DGFN, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        # 分组卷积
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)  # 在通道维度分为两部分
        x = F.gelu(x1) * x2 + F.gelu(x2) * x1  # 双门控机制
        x = self.project_out(x)
        return x


class SGTA(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=1, bias=False):
        super(SGTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.sr_ratio = sr_ratio  # 空间稀疏因子
        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            self.norm = LayerNorm(dim)

            self.qk = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
            self.qk_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

            self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
            self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        else:
            self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
            self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        if self.sr_ratio>1:
            qk = self.norm(self.sr(x))  # 对qk进行空间稀疏一次降低计算量
            _, _, h_qk, w_qk = qk.shape
            qk = self.qk_dwconv(self.qk(qk))
            q, k = qk.chunk(2, dim=1)
            v = self.v_dwconv(self.v(x))

            q = rearrange(q, 'b (head c) h_qk w_qk -> b head c (h_qk w_qk)', head=self.num_heads)
            k = rearrange(k, 'b (head c) h_qk w_qk -> b head c (h_qk w_qk)', head=self.num_heads)
            v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        else:
            qkv = self.qkv_dwconv(self.qkv(x))
            q, k, v = qkv.chunk(3, dim=1)

            q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 归一化
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 转置自注意力
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        x = (attn @ v)

        x = rearrange(x, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type=LayerNorm_type)
        self.attn = SGTA(dim, num_heads=num_heads, sr_ratio=sr_ratio, bias=bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type=LayerNorm_type)
        self.ffn = DGFN(dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

        layer_scale_init_value = 0.
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x))
        x = x + self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.ffn(self.norm2(x))
        return x