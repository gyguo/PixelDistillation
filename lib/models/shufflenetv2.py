#!/usr/bin/env python

"""
ShuffleNetV2. https://github.com/pytorch/vision/blob/master/torchvision/models/shufflenetv2.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from collections import OrderedDict

__all__ = [
    'ShuffleNetV2', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
]

model_urls = {
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenetv2_x1.5': None,
    'shufflenetv2_x2.0': None,
}

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, pre_act=True):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            if pre_act:
                self.branch1 = nn.Sequential(
                    nn.ReLU(inplace=False),
                    self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                    nn.BatchNorm2d(inp),
                    nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(branch_features),
                )
            else:
                self.branch1 = nn.Sequential(
                    self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                    nn.BatchNorm2d(inp),
                    nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(branch_features),
                )
        else:
            self.branch1 = nn.Sequential()

        if pre_act:
            self.branch2 = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(inp if (self.stride > 1) else branch_features,
                          branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
                self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(branch_features),
                nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
            )
        else:
            self.branch2 = nn.Sequential(
                nn.Conv2d(inp if (self.stride > 1) else branch_features,
                          branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
                self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(branch_features),
                nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
            )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, num_classes=1000, inverted_residual=InvertedResidual):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            pre_act = False if name == 'stage2' else True
            seq = [inverted_residual(input_channels, output_channels, 2, pre_act=pre_act)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.fc_cls = nn.Linear(output_channels, num_classes)

    def forward(self, x, return_feat=False, preReLU=False, return_final=False):
        feat_input = self.conv1(x)  # 56 x 56
        x = F.relu(feat_input)
        x = self.maxpool(x)

        feat1 = self.stage2(x)
        feat2 = self.stage3(feat1)
        feat3 = self.stage4(feat2)
        feat4 = self.conv5(feat3)

        x = F.adaptive_avg_pool2d(feat4, (1, 1))
        x = x.view(x.size(0), x.size(1))
        if return_final:
            finall_feat = x
        x = self.fc_cls(x)

        if return_feat:
            if not preReLU:
                feat_input = F.relu(feat_input)
                feat1 = F.relu(feat1)
                feat2 = F.relu(feat2)
                feat3 = F.relu(feat3)
                feat4 = F.relu(feat4)

            if return_final:
                return x, [feat_input, feat1, feat2, feat3, feat4, finall_feat]
            else:
                return x, [feat_input, feat1, feat2, feat3, feat4]
        else:
            return x

    def get_feat_size(self, x):
        """
        :param x: input
        :return: size of final feat
        """
        _, feats = self.forward(x, return_feat=True)
        feat_size = feats[-1].shape
        return list(feat_size)

    def get_input_feat_size(self, x):
        """
        :param x: input
        :return: size of input feat
        """
        _, feats = self.forward(x, return_feat=True)
        feat_size = feats[0].shape
        return list(feat_size)



def _shufflenetv2(arch, pretrained, progress, *args, **kwargs):
    model = ShuffleNetV2(*args, **kwargs)

    if pretrained:
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            _pretrained_dict = OrderedDict()
            for idx, (k, v) in enumerate(state_dict.items()):
                splitted_k = k.split('.')
                # special for 1.0x
                if 29 < idx < 280:
                    splitted_k[-2] = str(int(splitted_k[-2]) + 1)
                    _pretrained_dict['.'.join(splitted_k)] = v
                else:
                    _pretrained_dict[k] = v

            model_dict = model.state_dict()
            _pretrained_dict = {k: v for k, v in _pretrained_dict.items() if k in model_dict}
            model_dict.update(_pretrained_dict)
            model.load_state_dict(model_dict)
            # release
            del _pretrained_dict
            del state_dict
    return model

def shufflenet_v2_x0_5(pretrained=False, progress=True, **kwargs):
    return _shufflenetv2('shufflenetv2_x0.5', pretrained, progress,
                         [4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)

def shufflenet_v2_x1_0(pretrained=False, progress=True, **kwargs):
    return _shufflenetv2('shufflenetv2_x1.0', pretrained, progress,
                         [4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)


if __name__ == "__main__":

    input = torch.ones(1, 3, 56, 56)
    model = shufflenet_v2_x1_0(pretrained=True)
    x, feats = model(input, return_feat=True, preReLU=False, return_final=True)

    print(feats[0].shape)
    print(feats[1].shape)
    print(feats[2].shape)
    print(feats[3].shape)
    print(feats[4].shape)
    print(feats[5].shape)
