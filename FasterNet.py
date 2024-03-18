import torch
import torch.nn as nn
from base_network import LayerNorm


# 部分卷积
class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x):
        # 在通道维度进行分割，对其中的一部分进行卷积，而其余通道特征保持不变
        x1, x2 = x[:, :self.dim_conv3, :, :], x[:, self.dim_conv3:, :, :]
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)  # 通道维度拼接
        return x

import torch
import torch.nn as nn

class DConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DConv, self).__init__()
        # 深度卷积（depthwise convolution）
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        # 逐点卷积（pointwise convolution）
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 深度卷积
        x = self.depthwise(x)
        # 逐点卷积
        x = self.pointwise(x)
        return x


# 多层感知机块，部分卷积和两层1*1卷积
class MLPBlock(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=2,
                 layer_scale_init_value=0.,
                 act_layer=nn.ReLU,
                 norm_layer=LayerNorm
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio


        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False))

        self.spatial_mixing = DConv(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)  # 部分卷积
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        shortcut = x
        x = self.spatial_mixing(x)  # 部分卷积
        x = shortcut + self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x)
        return x


class FasterNetBlock(nn.Module):
    def __init__(self,
                 dim=48,
                 depth=2,
                 # n_div=4,  # 通道维度分割系数
                 mlp_ratio=2,  # 多层感知机扩张系数
                 layer_scale_init_value=0.,
                 norm_layer=LayerNorm,  # 归一化
                 act_layer=nn.ReLU  # 激活函数
                 ):
        super().__init__()

        self.blocks = nn.Sequential(*[
            MLPBlock(dim=dim,  mlp_ratio=mlp_ratio, layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer, act_layer=act_layer)for _ in range(depth)])

    def forward(self, x):
        x = self.blocks(x)
        return x



