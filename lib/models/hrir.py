import torch
import torch.nn as nn
import torch.nn.functional as F


class SR1x1(nn.Module):

    def __init__(self, cfg, feat_size):
        super(SR1x1, self).__init__()

        self.image_size = cfg.DATA.CROP_SIZE
        self.in_size = feat_size[2]
        if self.image_size % self.in_size == 0:
            self.scale_factor = int(self.image_size / self.in_size)
        else:
            self.scale_factor = int(self.image_size / self.in_size) +1

        self.outplanes = self.scale_factor ** 2 * 3
        self.inplanes = feat_size[1]

        self.conv = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)
        self.prelu = nn.PReLU(3)

    def forward(self, feat_s):

        feat = self.conv(feat_s)

        image_sr = self.pixel_shuffle(feat)

        image_sr = self.prelu(image_sr)

        if self.image_size % self.in_size == 0:
            return image_sr
        else:
            return image_sr[:, :, 0:self.image_size, 0:self.image_size]

if __name__ == "__main__":
    import os, sys

    sys.path.insert(0, '../../lib')

    import argparse
    from config.default import update_config
    from config.default import config as cfg

    parser = argparse.ArgumentParser(description='knowledge distillation')
    parser.add_argument('--config_file', type=str, default='../../configs/kd/cub/cub_resnet50_resnet_pixel.yaml',
                        required=False, help='Optional config file for params')
    parser.add_argument('opts', help='see config.py for all options',
                        default='BASIC.GPU_ID [0]', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(args)

    cfg.DATA.CROP_SIZE = 224
    cfg.DATA.LESSEN_RATIO = 4.0
    input_s = torch.randn(64, 3, 56, 56)
    feat_s = torch.randn(64, 64, 28, 28)
    feat_t = torch.randn(64, 64, 112, 112)
    weight_t = torch.randn(64, 3, 7, 7)

    model = FSR(cfg, list(feat_s.shape))
    image, feat = model(feat_s, weight_t)
    print(image.shape)
    print(feat.shape)