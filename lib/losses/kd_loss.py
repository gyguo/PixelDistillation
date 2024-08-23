import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillKL(torch.nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, temp):
        super(DistillKL, self).__init__()
        self.temp = temp

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.temp, dim=1)
        p_t = F.softmax(y_t/self.temp, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.temp**2) / y_s.shape[0]
        return loss


class KDLoss(nn.Module):
    def __init__(self, cfg):
        super(KDLoss, self).__init__()
        self.alpha = cfg.KD.ALPHA
        self.DistillKL = DistillKL(cfg.KD.TEMP)
        self.cls_criterion = torch.nn.CrossEntropyLoss()

    def forward(self, output_s, output_t, target):
        cls_loss = self.cls_criterion(output_s, target)
        kd_loss = self.DistillKL(output_s, output_t)
        loss = (1 - self.alpha) * cls_loss + self.alpha * kd_loss
        return loss