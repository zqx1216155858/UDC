import torch
import torch.nn as nn
from base_network import OverlapPatchEmbed, Downsample, Upsample
from Transformer import TransformerBlock
from FasterNet import FasterNetBlock
import math

class DLGNet(nn.Module):
    def __init__(self,
                 inp_channels=3,  # 输入通道数
                 out_channels=3,  # 输出通道数
                 dim=48,  # 输入层输出的通道维度
                 num_blocks=(2, 4, 4, 8),  # 每一编解码阶段的网络模块个数
                 heads=(1, 2, 4, 8),  # 编码阶段的自注意力头个数
                 sr_ratios=(4, 2, 2, 1),  # 编码阶段的空间稀疏因子
                 ffn_expansion_factor= math.exp(1),  # 前馈网络扩张因子
                 bias=False,  # 是否使用网络偏置
                 LayerNorm_type='WithBias',  # 层标准化的类型
                 ):
        super(DLGNet, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)  # patch embedding
        # 编码器

        self.encoder1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], sr_ratio=sr_ratios[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[0])])

        self.down1 = Downsample(dim)  # 下采样
        # 编码器
        self.encoder2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], sr_ratio=sr_ratios[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[1])])

        self.down2 = Downsample(int(dim * 2 ** 1))  # 下采样
        # 编码器
        self.encoder3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], sr_ratio=sr_ratios[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[2])])

        self.down3 = Downsample(int(dim * 2 ** 2))  # 下采样
        # 编码器
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], sr_ratio=sr_ratios[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[3])])

        self.up3 = Upsample(int(dim * 2 ** 3))  # 上采样
        self.reduce_chan_3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)  # 通道降维
        self.decoder3 = FasterNetBlock(dim=int(dim * 2 ** 2), depth=num_blocks[2])  # 解码器

        self.up2 = Upsample(int(dim * 2 ** 2))  # 上采样
        self.reduce_chan_2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)  # 通道降维
        self.decoder2 = FasterNetBlock(dim=int(dim * 2 ** 1), depth=num_blocks[1])  # 解码器

        self.up1 = Upsample(int(dim * 2 ** 1))  # 上采样
        self.reduce_chan_1 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 0), kernel_size=1, bias=bias)  # 通道降维
        self.decoder1 = FasterNetBlock(dim=int(dim * 2 ** 0), depth=num_blocks[0])  # 解码器

        # 输出层
        self.output = nn.Conv2d(int(dim * 2 ** 0), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        input_ = x
        # 编码器1
        x1 = self.patch_embed(x)
        # print(x1.shape,'before encoder')
        x1 = self.decoder1(x1) + x1
        # print(x1.shape,'after encoder')
        x1_skip = x1

        # 编码器2
        x2 = self.down1(x1)

        x2 = self.decoder2(x2) + x2
        x2_skip = x2

        # 编码器3
        x3 = self.down2(x2)
        x3 = self.decoder3(x3) + x3

        x3_skip = x3

        # 第四阶段
        x4 = self.down3(x3)
        x4 = self.latent(x4)
        x3 = self.up3(x4)

        # 解码器3
        x3 = torch.cat([x3, x3_skip], 1)  # 跳跃连接
        x3 = self.reduce_chan_3(x3)
        x3 = self.encoder3(x3)
        x2 = self.up2(x3)

        # 解码器2
        x2 = torch.cat([x2, x2_skip], 1)  # 跳跃连接
        x2 = self.reduce_chan_2(x2)
        x2 = self.encoder2(x2)
        x1 = self.up1(x2)

        # 解码器1
        x1 = torch.cat([x1, x1_skip], 1)  # 跳跃连接
        x1 = self.reduce_chan_1(x1)
        x1 = self.encoder1(x1)
        # x1 = self.refinement(x1)
        x = self.output(x1) + input_  # 全局残差连接

        return x




if __name__ == '__main__':
    x = torch.randn((1, 3, 64, 64))
    net = DLGNet()
    y = net(x)
    print(y.shape)

    # 计算参数和计算量的方法
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(net, (3, 64, 64), as_strings=True, print_per_layer_stat=False)
    print(f'flops:{flops}, params:{params}')


