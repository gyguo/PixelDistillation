"""
feature distillation loss in input compression stage, using upsampled feature maps
"""
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F


class ICFUPLoss(nn.Module):
    def __init__(self, cfg):
        super(ICFUPLoss, self).__init__()
        self.mse_criterion = nn.MSELoss()

    def forward(self, g_s, g_t):
        return [self.icfup_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def icfup_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        f_s = F.interpolate(f_s, size=(t_H, t_H), mode='bilinear', align_corners=True)
        return self.mse_criterion(f_s, f_t)



if __name__ == "__main__":
    import torch
    feat1 = torch.ones(2, 512, 4, 4)
    feat2 = torch.ones(2, 512, 28, 28)

    icfup_loss = ICFUPLoss(None)

    loss = icfup_loss([feat1], [feat2])

