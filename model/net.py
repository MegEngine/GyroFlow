# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2020 Liang Liu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, 
# publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
# subject to the following conditions:
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# This repo is licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import math
import collections

import megengine.module as nn
import megengine.functional as F

from model.nn_upsample import NeuralUpsampler, FlowMaskEstimator

from common.utils import flow_warp, upsample2d_flow_as


def conv(inp, out, k=3, s=1, d=1, isReLU=True):
    if isReLU:
        ret = nn.Sequential(nn.Conv2d(inp, out, k, s, padding=((k - 1) * d) // 2, dilation=d, bias=True), nn.LeakyReLU(0.1))
    else:
        ret = nn.Sequential(nn.Conv2d(inp, out, k, s, padding=((k - 1) * d) // 2, dilation=d, bias=True))
    return ret


class ContextNetwork(nn.Module):
    def __init__(self, ch_in):
        super(ContextNetwork, self).__init__()
        self.convs = nn.Sequential(conv(ch_in, 128, 3, 1, 1), conv(128, 128, 3, 1, 2), conv(128, 128, 3, 1, 4), conv(128, 96, 3, 1, 8),
                                   conv(96, 64, 3, 1, 16), conv(64, 32, 3, 1, 1), conv(32, 2, isReLU=False))

    def forward(self, x):
        return self.convs(x)


class FlowEstimator(nn.Module):
    def __init__(self, ch_in):
        super(FlowEstimator, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(128, 128)
        self.conv3 = conv(128 + 128, 96)
        self.conv4 = conv(96 + 128, 64)
        self.conv5 = conv(96 + 64, 32)
        # channels of the second last layer
        self.feat_dim = 32
        self.predict_flow = conv(64 + 32, 2, isReLU=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(F.concat([x1, x2], axis=1))
        x4 = self.conv4(F.concat([x2, x3], axis=1))
        x5 = self.conv5(F.concat([x3, x4], axis=1))
        flow = self.predict_flow(F.concat([x4, x5], axis=1))
        return x5, flow


class CostVolume(nn.Module):
    def __init__(self, d=4, *args, **kwargs):
        super(CostVolume, self).__init__()
        self.d = d
        self.out_dim = 2 * self.d + 1
        self.pad_size = self.d

    def forward(self, x1, x2):
        _, _, H, W = x1.shape

        x2 = F.nn.pad(x2, ((0, 0), (0, 0), (self.pad_size, self.pad_size), (self.pad_size, self.pad_size)))
        cv = []
        for i in range(self.out_dim):
            for j in range(self.out_dim):
                cost = x1 * x2[:, :, i:(i + H), j:(j + W)]
                cost = F.mean(cost, 1, keepdims=True)
                cv.append(cost)
        return F.concat(cv, 1)


class FeaturePyramidExtractor(nn.Module):
    def __init__(self, pyr_chans):
        super(FeaturePyramidExtractor, self).__init__()
        self.pyr_chans = pyr_chans
        self.convs = []
        for _, (ch_in, ch_out) in enumerate(zip(pyr_chans[:-1], pyr_chans[1:])):
            layer = nn.Sequential(conv(ch_in, ch_out, s=2), conv(ch_out, ch_out))
            self.convs.append(layer)

    def forward(self, x):
        feature_pyramid = []
        for conv in self.convs:
            x = conv(x)
            feature_pyramid.append(x)
        return feature_pyramid[::-1]


class GyroFlow(nn.Module):
    def __init__(self, params):
        super(GyroFlow, self).__init__()
        self.leakyRELU = nn.LeakyReLU(0.1)

        self.upsample = params.upsample
        self.with_bk = True

        self.pyr_chans = [3, 16, 32, 64, 96, 128, 192]
        self.feature_pyramid_extractor = FeaturePyramidExtractor(self.pyr_chans)
        # correlation range
        self.d = 4
        self.output_level = 4
        # cost volume
        self.cost_volume = CostVolume(d=self.d)
        self.cv_dim = (self.d * 2 + 1)**2
        self.upsampler = NeuralUpsampler()

        self.ch_inp = 32 + self.cv_dim + 2
        self.flow_estimator = FlowEstimator(self.ch_inp)
        self.context_net = ContextNetwork(self.flow_estimator.feat_dim + 2)

        self.conv_1x1 = list([
            conv(192, 32, k=1, s=1, d=1),
            conv(128, 32, k=1, s=1, d=1),
            conv(96, 32, k=1, s=1, d=1),
            conv(64, 32, k=1, s=1, d=1),
            conv(32, 32, k=1, s=1, d=1)
        ])

        self.with_gyro_field = True
        self.flow_predictor = FlowMaskEstimator(4, (8, 16, 32, 16, 8), 2)
        self.mask_predictor = FlowMaskEstimator(64, (32, 32, 32, 16, 8), 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.msra_normal_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init.calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def generate_fused_flow(self, x1, x2):
        input_feature = F.concat((x1, x2), axis=1)
        flow = self.flow_predictor(input_feature)[1]
        assert flow.shape[1] == 2
        return flow

    def generate_map(self, x1, x2):
        input_feature = F.concat((x1, x2), axis=1)
        out = self.mask_predictor(input_feature)[1]
        mask = F.sigmoid(out)
        assert mask.shape[1] == 1
        return mask

    def self_guided_fusion_module(self, flow, gyro_field_rsz, x1, x2_warp, layer):
        fuse_flow = self.generate_fused_flow(flow, gyro_field_rsz)
        mask = self.generate_map(self.conv_1x1[layer](x1), self.conv_1x1[layer](x2_warp))
        flow = fuse_flow * mask + gyro_field_rsz * (1 - mask)
        return flow

    def normalize_features(self, feature_list, normalize, center, moments_across_channels=True, moments_across_images=True):
        # Compute feature statistics.
        statistics = collections.defaultdict(list)
        axes = [1, 2, 3] if moments_across_channels else [2, 3]  # [b, c, h, w]
        for feature_image in feature_list:
            mean = F.mean(feature_image, axis=axes, keepdims=True)  # [b,1,1,1] or [b,c,1,1]
            variance = F.var(feature_image, axis=axes, keepdims=True)  # [b,1,1,1] or [b,c,1,1]
            statistics['mean'].append(mean)
            statistics['var'].append(variance)

        if moments_across_images:
            statistics['mean'] = ([F.mean(F.stack(statistics['mean'], axis=0), axis=(0, ))] * len(feature_list))
            statistics['var'] = ([F.var(F.stack(statistics['var'], axis=0), axis=(0, ))] * len(feature_list))

        statistics['std'] = [F.sqrt(v + 1e-16) for v in statistics['var']]

        # Center and normalize features.
        if center:
            feature_list = [f - mean for f, mean in zip(feature_list, statistics['mean'])]
        if normalize:
            feature_list = [f / std for f, std in zip(feature_list, statistics['std'])]
        return feature_list

    def predict_flow(self, x1_pyrs, x2_pyrs, gyro_field=None):
        flow_pyrs = []

        batch_size, _, h_x1, w_x1 = x1_pyrs[0].shape
        dtype = x1_pyrs[0].dtype

        flow = F.zeros((batch_size, 2, h_x1, w_x1), dtype=dtype)

        for layer, (x1, x2) in enumerate(zip(x1_pyrs, x2_pyrs)):
            if layer == 0:
                x2_warp = x2
            else:
                flow = self.upsampler(flow, self.conv_1x1[layer](x1), self.conv_1x1[layer](x2))

                gyro_field_rsz = upsample2d_flow_as(gyro_field, flow, if_rate=True)
                x2_warp = flow_warp(x2, gyro_field_rsz)

                flow = self.self_guided_fusion_module(flow, gyro_field_rsz, x1, x2_warp, layer)

                x2_warp = flow_warp(x2, flow)

            # cost volume normalized
            x1_normalized, x2_warp_normalized = self.normalize_features([x1, x2_warp],
                                                                        normalize=True,
                                                                        center=True,
                                                                        moments_across_channels=False,
                                                                        moments_across_images=False)

            _cv = self.cost_volume(x1_normalized, x2_warp_normalized)
            _cv_relu = self.leakyRELU(_cv)

            x1 = self.conv_1x1[layer](x1)
            _x_feat, flow_pred = self.flow_estimator(F.concat([_cv_relu, x1, flow], axis=1))
            flow += flow_pred

            flow_refine = self.context_net(F.concat([_x_feat, flow], axis=1))
            flow += flow_refine

            flow_pyrs.append(flow)
            if layer == self.output_level:
                break
        if self.upsample:
            flows = [F.vision.interpolate(flow * 4, scale_factor=4, mode='bilinear', align_corners=True) for flow in flow_pyrs]
        return flows[::-1]

    def forward(self, data_batch, with_bk=True):
        x = data_batch['imgs']
        imgs = [x[:, 3 * i:3 * i + 3] for i in range(2)]
        x = [self.feature_pyramid_extractor(img) + [img] for img in imgs]

        gyro_field = data_batch["gyro_field"]

        res = {}
        res['flow_fw'] = self.predict_flow(x[0], x[1], gyro_field)
        if with_bk:
            res['flow_bw'] = self.predict_flow(x[1], x[0], -1 * gyro_field)
        return res


class GyroFlowTestFlops(GyroFlow):
    def forward(self, data_batch, with_bk=True):
        x = data_batch
        imgs = [x[:, 3 * i:3 * i + 3] for i in range(2)]
        x = [self.feature_pyramid_extractor(img) + [img] for img in imgs]

        gyro_field = F.ones_like(data_batch)[:, :2, ...]

        res_fw = self.predict_flow(x[0], x[1], gyro_field)
        if with_bk:
            res_bw = self.predict_flow(x[1], x[0], -1 * gyro_field)
        return res_fw, res_bw


def fetch_net(params):
    if params.net_type == "gyroflow":
        net = GyroFlow(params)
    else:
        raise NotImplementedError
    return net
