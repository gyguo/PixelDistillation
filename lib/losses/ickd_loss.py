from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2, dim = 1):
        super(Normalize, self).__init__()
        self.power = power
        self.dim = dim

    def forward(self, x):
        norm = x.pow(self.power).sum(self.dim, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Embed(nn.Module):
    def __init__(self, dim_in=256, dim_out=128):
        super(Embed, self).__init__()
        self.conv2d = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.l2norm = nn.BatchNorm2d(dim_out)#Normalize(2)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.l2norm(x)
        return x


class ICKDLoss(nn.Module):
    """Inter-Channel Correlation"""
    def __init__(self, cfg):
        super(ICKDLoss, self).__init__()
        self.embed_s = Embed(cfg.ICKD.FEATDIM_S[1], cfg.ICKD.FEATDIM_T[1])
        self.embed_t = Embed(cfg.ICKD.FEATDIM_S[1], cfg.ICKD.FEATDIM_T[1])

    def forward(self, g_s, g_t):
        loss =  [self.batch_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]
        return loss

    def batch_loss(self, f_s, f_t):
        f_s = self.embed_s(f_s)
        bsz, ch = f_s.shape[0], f_s.shape[1]

        f_s = f_s.view(bsz, ch, -1)
        f_t = f_t.view(bsz, ch, -1)

        emd_s = torch.bmm(f_s, f_s.permute(0,2,1))
        emd_s = torch.nn.functional.normalize(emd_s, dim = 2)

        emd_t = torch.bmm(f_t, f_t.permute(0,2,1))
        emd_t = torch.nn.functional.normalize(emd_t, dim = 2)

        G_diff = emd_s - emd_t
        loss = (G_diff * G_diff).view(bsz, -1).sum() / (ch*bsz)
        return loss


if __name__ == "__main__":
    x1 = torch.randn(2, 64, 112, 112)
    x2 = torch.randn(2, 32, 64, 64)
    s_dim = x1.shape[1]
    feat_dim = x2.shape[1]
    kd = ICKDLoss(s_dim, feat_dim)
    kd_loss = kd([x1], [x2])
    print(kd_loss)