# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2021 coolbeam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# This repo is licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import megengine.module as nn
import megengine.functional as F

from common.utils import flow_warp, upsample2d_flow_as


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True, if_IN=False, IN_affine=False, if_BN=False):
    if isReLU:
        if if_IN:
            return nn.Sequential(
                nn.Conv2d(in_planes,
                          out_planes,
                          kernel_size=kernel_size,
                          stride=stride,
                          dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2,
                          bias=True), nn.LeakyReLU(0.1), nn.InstanceNorm(out_planes, affine=IN_affine))
        elif if_BN:
            return nn.Sequential(
                nn.Conv2d(in_planes,
                          out_planes,
                          kernel_size=kernel_size,
                          stride=stride,
                          dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2,
                          bias=True), nn.LeakyReLU(0.1), nn.BatchNorm2d(out_planes, affine=IN_affine))
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes,
                          out_planes,
                          kernel_size=kernel_size,
                          stride=stride,
                          dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2,
                          bias=True), nn.LeakyReLU(0.1))
    else:
        if if_IN:
            return nn.Sequential(
                nn.Conv2d(in_planes,
                          out_planes,
                          kernel_size=kernel_size,
                          stride=stride,
                          dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2,
                          bias=True), nn.InstanceNorm(out_planes, affine=IN_affine))
        elif if_BN:
            return nn.Sequential(
                nn.Conv2d(in_planes,
                          out_planes,
                          kernel_size=kernel_size,
                          stride=stride,
                          dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2,
                          bias=True), nn.BatchNorm2d(out_planes, affine=IN_affine))
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes,
                          out_planes,
                          kernel_size=kernel_size,
                          stride=stride,
                          dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2,
                          bias=True))


class FlowEstimatorDense_temp(nn.Module):
    def __init__(self, ch_in=64, f_channels=(128, 128, 96, 64, 32, 32), ch_out=2):
        super(FlowEstimatorDense_temp, self).__init__()

        N = 0
        ind = 0
        N += ch_in

        self.conv1 = conv(N, f_channels[ind])
        N += f_channels[ind]
        ind += 1

        self.conv2 = conv(N, f_channels[ind])
        N += f_channels[ind]
        ind += 1

        self.conv3 = conv(N, f_channels[ind])
        N += f_channels[ind]
        ind += 1

        self.conv4 = conv(N, f_channels[ind])
        N += f_channels[ind]
        ind += 1

        self.conv5 = conv(N, f_channels[ind])
        N += f_channels[ind]
        self.num_feature_channel = N
        ind += 1

        self.conv_last = conv(N, ch_out, isReLU=False)

    def forward(self, x):
        x1 = F.concat([self.conv1(x), x], axis=1)
        x2 = F.concat([self.conv2(x1), x1], axis=1)
        x3 = F.concat([self.conv3(x2), x2], axis=1)
        x4 = F.concat([self.conv4(x3), x3], axis=1)
        x5 = F.concat([self.conv5(x4), x4], axis=1)
        x_out = self.conv_last(x5)
        return x5, x_out


class FlowMaskEstimator(FlowEstimatorDense_temp):
    def __init__(self, ch_in, f_channels, ch_out):
        super(FlowMaskEstimator, self).__init__(ch_in=ch_in, f_channels=f_channels, ch_out=ch_out)


class NeuralUpsampler(nn.Module):
    def __init__(self):
        super(NeuralUpsampler, self).__init__()
        f_channels_es = (32, 32, 32, 16, 8)
        in_C = 64
        self.dense_estimator_mask = FlowEstimatorDense_temp(in_C, f_channels=f_channels_es, ch_out=3)
        self.upsample_output_conv = nn.Sequential(
            conv(3, 16, kernel_size=3, stride=1, dilation=1),
            conv(16, 16, stride=2),
            conv(16, 32, kernel_size=3, stride=1, dilation=1),
            conv(32, 32, stride=2),
        )

    def forward(self, flow_init, feature_1, feature_2, output_level_flow=None):
        n, c, h, w = flow_init.shape
        n_f, c_f, h_f, w_f = feature_1.shape

        if h != h_f or w != w_f:
            flow_init = F.vision.interpolate(flow_init, scale_factor=2., mode='bilinear', align_corners=True) * 2
        feature_2_warp = flow_warp(feature_2, flow_init)
        input_feature = F.concat((feature_1, feature_2_warp), axis=1)
        _, x_out = self.dense_estimator_mask(input_feature)
        inter_flow = x_out[:, :2, :, :]
        inter_mask = x_out[:, 2, :, :]

        inter_mask = F.expand_dims(inter_mask, 1)
        inter_mask = F.sigmoid(inter_mask)

        if output_level_flow is not None:
            inter_flow = upsample2d_flow_as(inter_flow, output_level_flow, mode="bilinear", if_rate=True)
            inter_mask = upsample2d_flow_as(inter_mask, output_level_flow, mode="bilinear")
            flow_init = output_level_flow

        flow_up = flow_warp(flow_init, inter_flow) * (1 - inter_mask) + flow_init * inter_mask
        return flow_up

    def output_conv(self, x):
        return self.upsample_output_conv(x)
