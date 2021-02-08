# -*- coding: utf-8 -*-

__author__ = 'kohou.wang'
__time__ = '19-11-20'
__email__ = 'oukohou@outlook.com'

# If this runs wrong, don't ask me, I don't know why;
# If this runs right, thank god, and I don't know why.
# Maybe the answer, my friend, is blowing in the wind.
# Well, I'm kidding... Always, Welcome to contact me.

"""Description for the script:
MixNet model in pytorch.

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return x * self.sigmoid(x)


NON_LINEARITY = {
    'ReLU': nn.ReLU(inplace=True),
    'Swish': Swish(),
}


def _RoundChannels(c, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_c = max(min_value, int(c + divisor / 2) // divisor * divisor)
    if new_c < 0.9 * c:
        new_c += divisor
    return new_c


def _SplitChannels(channels, num_groups):
    split_channels = [channels // num_groups for _ in range(num_groups)]
    split_channels[0] += channels - sum(split_channels)
    return split_channels


def Conv3x3Bn(in_channels, out_channels, stride, non_linear='ReLU'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        NON_LINEARITY[non_linear]
    )


def Conv1x1Bn(in_channels, out_channels, non_linear='ReLU'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        NON_LINEARITY[non_linear]
    )


class SqueezeAndExcite(nn.Module):
    def __init__(self, channels, squeeze_channels, se_ratio):
        super(SqueezeAndExcite, self).__init__()
        
        squeeze_channels = squeeze_channels * se_ratio
        if not squeeze_channels.is_integer():
            raise ValueError('channels must be divisible by 1/ratio')
        
        squeeze_channels = int(squeeze_channels)
        self.se_reduce = nn.Conv2d(channels, squeeze_channels, 1, 1, 0, bias=True)
        self.non_linear1 = NON_LINEARITY['Swish']
        self.se_expand = nn.Conv2d(squeeze_channels, channels, 1, 1, 0, bias=True)
        self.non_linear2 = nn.Sigmoid()
    
    def forward(self, x):
        y = torch.mean(x, (2, 3), keepdim=True)
        y = self.non_linear1(self.se_reduce(y))
        y = self.non_linear2(self.se_expand(y))
        y = x * y
        
        return y


class GroupedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(GroupedConv2d, self).__init__()
        
        self.num_groups = len(kernel_size)
        self.split_in_channels = _SplitChannels(in_channels, self.num_groups)
        self.split_out_channels = _SplitChannels(out_channels, self.num_groups)
        
        self.grouped_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.grouped_conv.append(nn.Conv2d(
                self.split_in_channels[i],
                self.split_out_channels[i],
                kernel_size[i],
                stride=stride,
                padding=padding,
                bias=False
            ))
    
    def forward(self, x):
        if self.num_groups == 1:
            return self.grouped_conv[0](x)
        
        x_split = torch.split(x, self.split_in_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.grouped_conv, x_split)]
        x = torch.cat(x, dim=1)
        
        return x


class MDConv(nn.Module):
    def __init__(self, channels, kernel_size, stride):
        super(MDConv, self).__init__()
        
        self.num_groups = len(kernel_size)
        self.split_channels = _SplitChannels(channels, self.num_groups)
        
        self.mixed_depthwise_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.mixed_depthwise_conv.append(nn.Conv2d(
                self.split_channels[i],
                self.split_channels[i],
                kernel_size[i],
                stride=stride,
                padding=kernel_size[i] // 2,
                groups=self.split_channels[i],
                bias=False
            ))
    
    def forward(self, x):
        if self.num_groups == 1:
            return self.mixed_depthwise_conv[0](x)
        
        x_split = torch.split(x, self.split_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.mixed_depthwise_conv, x_split)]
        x = torch.cat(x, dim=1)
        
        return x


class MixNetBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=[3],
            expand_ksize=[1],
            project_ksize=[1],
            stride=1,
            expand_ratio=1,
            non_linear='ReLU',
            se_ratio=0.0
    ):
        
        super(MixNetBlock, self).__init__()
        
        expand = (expand_ratio != 1)
        expand_channels = in_channels * expand_ratio
        se = (se_ratio != 0.0)
        self.residual_connection = (stride == 1 and in_channels == out_channels)
        
        conv = []
        
        if expand:
            # expansion phase
            pw_expansion = nn.Sequential(
                GroupedConv2d(in_channels, expand_channels, expand_ksize),
                nn.BatchNorm2d(expand_channels),
                NON_LINEARITY[non_linear]
            )
            conv.append(pw_expansion)
        
        # depthwise convolution phase
        dw = nn.Sequential(
            MDConv(expand_channels, kernel_size, stride),
            nn.BatchNorm2d(expand_channels),
            NON_LINEARITY[non_linear]
        )
        conv.append(dw)
        
        if se:
            # squeeze and excite
            squeeze_excite = SqueezeAndExcite(expand_channels, in_channels, se_ratio)
            conv.append(squeeze_excite)
        
        # projection phase
        pw_projection = nn.Sequential(
            GroupedConv2d(expand_channels, out_channels, project_ksize),
            nn.BatchNorm2d(out_channels)
        )
        conv.append(pw_projection)
        
        self.conv = nn.Sequential(*conv)
    
    def forward(self, x):
        if self.residual_connection:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MixNet(nn.Module):
    # [in_channels, out_channels, kernel_size, expand_ksize, project_ksize, stride, expand_ratio, non_linear, se_ratio]
    mixnet_s = [(16, 16, [3], [1], [1], 1, 1, 'ReLU', 0.0),
                (16, 24, [3], [1, 1], [1, 1], 2, 6, 'ReLU', 0.0),
                (24, 24, [3], [1, 1], [1, 1], 1, 3, 'ReLU', 0.0),
                (24, 40, [3, 5, 7], [1], [1], 2, 6, 'Swish', 0.5),
                (40, 40, [3, 5], [1, 1], [1, 1], 1, 6, 'Swish', 0.5),
                (40, 40, [3, 5], [1, 1], [1, 1], 1, 6, 'Swish', 0.5),
                (40, 40, [3, 5], [1, 1], [1, 1], 1, 6, 'Swish', 0.5),
                (40, 80, [3, 5, 7], [1], [1, 1], 2, 6, 'Swish', 0.25),
                (80, 80, [3, 5], [1], [1, 1], 1, 6, 'Swish', 0.25),
                (80, 80, [3, 5], [1], [1, 1], 1, 6, 'Swish', 0.25),
                (80, 120, [3, 5, 7], [1, 1], [1, 1], 1, 6, 'Swish', 0.5),
                (120, 120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, 'Swish', 0.5),
                (120, 120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, 'Swish', 0.5),
                (120, 200, [3, 5, 7, 9, 11], [1], [1], 2, 6, 'Swish', 0.5),
                (200, 200, [3, 5, 7, 9], [1], [1, 1], 1, 6, 'Swish', 0.5),
                (200, 200, [3, 5, 7, 9], [1], [1, 1], 1, 6, 'Swish', 0.5)]
    
    mixnet_m = [(24, 24, [3], [1], [1], 1, 1, 'ReLU', 0.0),
                (24, 32, [3, 5, 7], [1, 1], [1, 1], 2, 6, 'ReLU', 0.0),
                (32, 32, [3], [1, 1], [1, 1], 1, 3, 'ReLU', 0.0),
                (32, 40, [3, 5, 7, 9], [1], [1], 2, 6, 'Swish', 0.5),
                (40, 40, [3, 5], [1, 1], [1, 1], 1, 6, 'Swish', 0.5),
                (40, 40, [3, 5], [1, 1], [1, 1], 1, 6, 'Swish', 0.5),
                (40, 40, [3, 5], [1, 1], [1, 1], 1, 6, 'Swish', 0.5),
                (40, 80, [3, 5, 7], [1], [1], 2, 6, 'Swish', 0.25),
                (80, 80, [3, 5, 7, 9], [1, 1], [1, 1], 1, 6, 'Swish', 0.25),
                (80, 80, [3, 5, 7, 9], [1, 1], [1, 1], 1, 6, 'Swish', 0.25),
                (80, 80, [3, 5, 7, 9], [1, 1], [1, 1], 1, 6, 'Swish', 0.25),
                (80, 120, [3], [1], [1], 1, 6, 'Swish', 0.5),
                (120, 120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, 'Swish', 0.5),
                (120, 120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, 'Swish', 0.5),
                (120, 120, [3, 5, 7, 9], [1, 1], [1, 1], 1, 3, 'Swish', 0.5),
                (120, 200, [3, 5, 7, 9], [1], [1], 2, 6, 'Swish', 0.5),
                (200, 200, [3, 5, 7, 9], [1], [1, 1], 1, 6, 'Swish', 0.5),
                (200, 200, [3, 5, 7, 9], [1], [1, 1], 1, 6, 'Swish', 0.5),
                (200, 200, [3, 5, 7, 9], [1], [1, 1], 1, 6, 'Swish', 0.5)]
    
    def __init__(self, net_type='mixnet_s', input_size=224, num_classes=1000, stem_channels=16, feature_size=1536, depth_multiplier=1.0):
        super(MixNet, self).__init__()
        
        if net_type == 'mixnet_s':
            config = self.mixnet_s
            stem_channels = 16
            dropout_rate = 0.2
        elif net_type == 'mixnet_m':
            config = self.mixnet_m
            stem_channels = 24
            dropout_rate = 0.25
        elif net_type == 'mixnet_l':
            config = self.mixnet_m
            stem_channels = 24
            depth_multiplier *= 1.3
            dropout_rate = 0.25
        else:
            raise TypeError('Unsupported MixNet type')
        
        assert input_size % 32 == 0
        
        # depth multiplier
        if depth_multiplier != 1.0:
            stem_channels = _RoundChannels(stem_channels * depth_multiplier)
            
            for i, conf in enumerate(config):
                conf_ls = list(conf)
                conf_ls[0] = _RoundChannels(conf_ls[0] * depth_multiplier)
                conf_ls[1] = _RoundChannels(conf_ls[1] * depth_multiplier)
                config[i] = tuple(conf_ls)
        
        # stem convolution
        self.stem_conv = Conv3x3Bn(3, stem_channels, 2)
        
        # building MixNet blocks
        layers = []
        for in_channels, out_channels, kernel_size, expand_ksize, project_ksize, stride, expand_ratio, non_linear, se_ratio in config:
            layers.append(MixNetBlock(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                expand_ksize=expand_ksize,
                project_ksize=project_ksize,
                stride=stride,
                expand_ratio=expand_ratio,
                non_linear=non_linear,
                se_ratio=se_ratio
            ))
        self.layers = nn.Sequential(*layers)
        
        # last several layers
        self.conv5 = nn.Conv2d(200, 32, 3, 2, )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(3 * 3 * 32, feature_size)
        self.fc2 = nn.Linear(feature_size, num_classes)
        
        self._initialize_weights()
    
    def forward(self, image_):
        # import ipdb
        # ipdb.set_trace()
        
        # 左右翻转
        # image_1 = torch.flip(image_, [3])
        # image_ = torch.cat((image_, image_1), dim=0)
        
        x = self.stem_conv(image_)
        x = self.layers(x)
        # import ipdb
        # ipdb.set_trace()
        x_conv5 = self.conv5(x)
        x_conv5 = self.dropout(x_conv5)
        x = x_conv5.view(x_conv5.size(0), -1)
        feat_ = self.fc1(x)
        
        # # # for 2 images
        # shape_ = feat_.shape[0]
        # feat_0, feat_1 = torch.split(feat_, shape_ // 2, 0)
        # feat_cat = torch.cat((feat_0, feat_1), dim=1)
        
        pred = self.fc2(feat_)
        # # for 2 images
        # pred_shape_ = pred.shape[0]
        # pred_0, pred_1 = torch.split(pred, pred_shape_ // 2, 0)
        #
        # pred_average = (pred_0 + pred_1) / 2
        
        return feat_, pred
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    net = MixNet('mixnet_s', feature_size=54, num_classes=5)
    x_image = torch.randn(1, 3, 224, 224)
    feat, preds = net(x_image)
    print(preds.shape)
    print(feat.shape)
