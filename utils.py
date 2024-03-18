import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


# 打补丁
def pad(x, factor=16, mode='reflect'):
    _, _, h_even, w_even = x.shape
    padh_left = (factor - h_even % factor) // 2
    padw_top = (factor - w_even % factor) // 2
    padh_right = padh_left if h_even % 2 == 0 else padh_left + 1  # 如果原图分辨率是奇数，则打补丁右边和下边多一个像素
    padw_bottom = padw_top if w_even % 2 == 0 else padw_top + 1
    x = F.pad(x, pad=[padw_top, padw_bottom, padh_left, padh_right], mode=mode)
    return x, (padh_left, padh_right, padw_top, padw_bottom)


# 打补丁逆向
def unpad(x, pad_size):
    padh_left, padh_right, padw_top, padw_bottom = pad_size
    _, _, newh, neww = x.shape
    h_start = padh_left
    h_end = newh - padh_right
    w_start = padw_top
    w_end = neww - padw_bottom
    x = x[:, :, h_start:h_end, w_start:w_end]  # 切片
    return x


def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps


class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()