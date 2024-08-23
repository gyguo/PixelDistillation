import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


# init for different models
def build_model(arch, num_classes, pretrained):
    if num_classes!=1000:
        from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d
        from .shufflenetv2 import shufflenet_v2_x0_5, shufflenet_v2_x1_0
    from .vit_pixel import vit_base_patch16_224, vit_tiny_patch16_224, vit_tiny_patch16_112, vit_tiny_patch16_56, \
        vit_small_patch16_112, vit_small_patch16_56

    model_dict = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152,
        'resnext50_32x4d': resnext50_32x4d,
        'shufflenet_v2_x0_5': shufflenet_v2_x0_5,
        'shufflenet_v2_x1_0': shufflenet_v2_x1_0,
        "vit_tiny_patch16_224": vit_tiny_patch16_224,
        "vit_tiny_patch16_112": vit_tiny_patch16_112,
        "vit_tiny_patch16_56": vit_tiny_patch16_56,
        "vit_base_patch16_224": vit_base_patch16_224,
        "vit_small_patch16_112": vit_small_patch16_112,
        "vit_small_patch16_56": vit_small_patch16_56
    }

    if arch not in model_dict:
        raise NotImplementedError('The model {} is not implemented!'.format(arch))
    elif arch in ['vit_tiny_patch16_224', 'vit_base_patch16_224']:
        from timm.models import create_model
        model = create_model(
        arch,
        pretrained=pretrained,
        num_classes=num_classes)
        return model
    else:
        return model_dict[arch](num_classes=num_classes, pretrained=pretrained)


class NaiveKDModelBuilder(nn.Module):
    def __init__(self, model_s, model_t):
        super(NaiveKDModelBuilder, self).__init__()
        self.model_s = model_s
        self.model_t = model_t

        # freeze the teacher model
        for p in self.model_t.parameters():
            p.requires_grad = False

    def forward(self, input1, input2=None, return_feat=False, preReLU=False, return_final=False):
        if input2 is None:
            input_small = input1
        else:
            input_small = input2

        if return_feat:
            logits_s, feats_s = self.model_s(input_small, return_feat=return_feat, preReLU=preReLU, return_final=return_final)
        else:
            logits_s = self.model_s(input_small)

        if self.training == False:
            return logits_s
        else:
            if return_feat:
                with torch.no_grad():
                    logits_t, feats_t = self.model_t(input1, return_feat=return_feat, preReLU=preReLU, return_final=return_final)
                return logits_s, feats_s, logits_t, feats_t
            else:
                with torch.no_grad():
                    logits_t = self.model_t(input1)
                return logits_s, logits_t


class ISRKDModelBuilder(nn.Module):
    def __init__(self, model_s, model_t, model_sr, position):
        super(ISRKDModelBuilder, self).__init__()
        self.model_s = model_s
        self.model_t = model_t
        self.model_sr = model_sr
        self.position = position

        # freeze the teacher model
        for p in self.model_t.parameters():
            p.requires_grad = False

    def forward(self, input1, input2=None, preReLU=False, return_final=False):
        if input2 is None:
            input_small = input1
        else:
            input_small = input2

        logits_s, feats_s = self.model_s(input_small, return_feat=True,
                                         preReLU=preReLU, return_final=return_final)

        if self.training == False:
            return logits_s
        else:
            image_sr = self.model_sr(feats_s[self.position])
            with torch.no_grad():
                logits_t = self.model_t(input1, return_feat=False, preReLU=preReLU, return_final=return_final)
            return logits_s, image_sr, logits_t


class ISRKDModelBuilderEval(nn.Module):
    def __init__(self, model_s, model_t, model_sr, position):
        super(ISRKDModelBuilderEval, self).__init__()
        self.model_s = model_s
        self.model_t = model_t
        self.model_sr = model_sr
        self.position = position

        # freeze the teacher model
        for p in self.model_t.parameters():
            p.requires_grad = False

    def forward(self, input1, input2=None, preReLU=False, return_final=False):
        if input2 is None:
            input_small = input1
        else:
            input_small = input2

        logits_s, feats_s = self.model_s(input_small, return_feat=True,
                                         preReLU=preReLU, return_final=return_final)

        image_sr = self.model_sr(feats_s[self.position])
        if self.training == False:
            return logits_s, image_sr
        else:
            with torch.no_grad():
                logits_t = self.model_t(input1, return_feat=False, preReLU=preReLU, return_final=return_final)
            return logits_s, image_sr, logits_t


class ISRKD_FKDModelBuilder(nn.Module):
    def __init__(self, model_s, model_t, model_sr, position):
        super(ISRKD_FKDModelBuilder, self).__init__()
        self.model_s = model_s
        self.model_t = model_t
        self.model_sr = model_sr
        self.position = position

        # freeze the teacher model
        for p in self.model_t.parameters():
            p.requires_grad = False

    def forward(self, input1, input2=None, preReLU=False, return_final=False):
        if input2 is None:
            input_small = input1
        else:
            input_small = input2

        logits_s, feats_s = self.model_s(input_small, return_feat=True,
                                         preReLU=preReLU, return_final=return_final)

        if self.training == False:
            return logits_s
        else:
            image_sr = self.model_sr(feats_s[self.position])
            with torch.no_grad():
                logits_t, feats_t = self.model_t(input1, return_feat=True, preReLU=preReLU, return_final=return_final)
            return logits_s, feats_s, image_sr, logits_t, feats_t


if __name__ == "__main__":
    import os, sys

    sys.path.insert(0, '../../lib')

    import argparse
    from config.default import update_config
    from config.default import config as cfg

    parser = argparse.ArgumentParser(description='knowledge distillation')
    parser.add_argument('--config_file', type=str, default='../../configs/cub/cub_resnet50_resnet_kd.yaml',
                        required=False, help='Optional config file for params')
    parser.add_argument('opts', help='see config.py for all options',
                        default='BASIC.GPU_ID [7]', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(args)

    cfg.BASIC.ROOT_DIR = os.path.join(os.path.dirname(__file__), '..', '..')

    import torch

    # fsr = FSRModelBuilder(cfg=cfg)
    # input = torch.ones(1, 3, 56, 56)
    # logits_s_lr, feats_s_lr, logits_s_hr, feat_s_hr = fsr(input)
    # print(feats_s_lr[-1].shape)
    # print(feat_s_hr.shape)

