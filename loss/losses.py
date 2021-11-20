
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
import numpy as np
import megengine as mge

import megengine.module as nn
import megengine.functional as F

from common.utils import flow_warp, upsample2d_flow_as, flow_error_avg


class LossL1(nn.Module):
    def __init__(self):
        super(LossL1, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, input, target):
        return self.loss(input, target)


class LossL2(nn.Module):
    def __init__(self):
        super(LossL2, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, input, target):
        return self.loss(input, target)


class LossSmoothL1(nn.Module):
    def __init__(self):
        super(LossSmoothL1, self).__init__()
        self.loss = nn.SmoothL1Loss()

    def forward(self, input, target):
        return self.loss(input, target)


class LossCrossEntropy(nn.Module):
    def __init__(self):
        super(LossCrossEntropy, self).__init__()
        self.loss = F.nn.cross_entropy

    def forward(self, input, target):
        return self.loss(input, target)


def _weighted_ssim(x, y, weight, c1=float('inf'), c2=9e-6, weight_epsilon=0.01):
    def _avg_pool3x3(x):
        xx = F.avg_pool2d(x, kernel_size=3, stride=1)
        return xx

    if c1 == float('inf') and c2 == float('inf'):
        raise ValueError('Both c1 and c2 are infinite, SSIM loss is zero. This is ' 'likely unintended.')
    average_pooled_weight = _avg_pool3x3(weight)
    weight_plus_epsilon = weight + weight_epsilon
    inverse_average_pooled_weight = 1.0 / (average_pooled_weight + weight_epsilon)

    def weighted_avg_pool3x3(z):
        wighted_avg = _avg_pool3x3(z * weight_plus_epsilon)
        return wighted_avg * inverse_average_pooled_weight

    mu_x = weighted_avg_pool3x3(x)
    mu_y = weighted_avg_pool3x3(y)
    sigma_x = weighted_avg_pool3x3(x**2) - mu_x**2
    sigma_y = weighted_avg_pool3x3(y**2) - mu_y**2
    sigma_xy = weighted_avg_pool3x3(x * y) - mu_x * mu_y
    if c1 == float('inf'):
        ssim_n = (2 * sigma_xy + c2)
        ssim_d = (sigma_x + sigma_y + c2)
    elif c2 == float('inf'):
        ssim_n = 2 * mu_x * mu_y + c1
        ssim_d = mu_x**2 + mu_y**2 + c1
    else:
        ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    result = ssim_n / ssim_d
    return F.clip((1 - result) / 2, 0, 1), average_pooled_weight


def photo_loss_ssim(im_x, im_y, occ_mask=None):
    if occ_mask is None:
        occ_mask = F.ones_like(im_x)
    loss_diff, occ_weight = _weighted_ssim(im_x, im_y, occ_mask)
    photo_loss = F.sum(loss_diff * occ_weight) / (F.sum(occ_weight) + 1e-6)
    return photo_loss


def SSIM(x, y, md=1):
    patch_size = 2 * md + 1
    C1 = 0.01**2
    C2 = 0.03**2

    mu_x = nn.AvgPool2d(patch_size, 1, 0, mode="average")(x)
    mu_y = nn.AvgPool2d(patch_size, 1, 0, mode="average")(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = F.pow(mu_x, 2)
    mu_y_sq = F.pow(mu_y, 2)

    sigma_x = nn.AvgPool2d(patch_size, 1, 0, mode="average")(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(patch_size, 1, 0, mode="average")(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(patch_size, 1, 0, mode="average")(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    dist = F.clip((1 - SSIM) / 2, 0, 1)
    return dist


def _photo_loss_census(img1, img2_warp, occu_mask1=None, max_distance=3, loss_type='L1', if_mask=False):
    patch_size = 2 * max_distance + 1

    def _ternary_transform_mge(image):
        n, c, h, w = image.shape
        if c == 3:
            R, G, B = F.split(image, 3, 1)
            intensities = (0.2989 * R + 0.5870 * G + 0.1140 * B)  # * 255  # convert to gray
        elif c == 1:
            intensities = image
        else:
            raise ValueError('image channel should be 3 or 1: %s' % c)
        # intensities = tf.image.rgb_to_grayscale(image) * 255
        out_channels = patch_size * patch_size
        w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))  # h,w,1,out_c
        w_ = np.transpose(w, (3, 2, 0, 1))  # 1,out_c,h,w
        # weight = torch.from_numpy(w_).float()
        weight = mge.tensor(w_.astype(np.float32))  # need check cuda?

        # if image.is_cuda:
        #     weight = weight.cuda()
        # patches_torch = torch.conv2d(input=out_channels, weight=weight, bias=None, stride=[1, 1], padding=[max_distance, max_distance])
        patches_mge = F.nn.conv2d(inp=intensities, weight=weight, bias=None, stride=[1, 1], padding=[max_distance, max_distance])
        transf_mge = patches_mge - intensities
        transf_norm_mge = transf_mge / F.sqrt(0.81 + transf_mge**2)
        return transf_norm_mge

    def _hamming_distance_mge(t1, t2):
        dist = (t1 - t2)**2
        dist = F.sum(dist / (0.1 + dist), axis=1, keepdims=True)
        return dist

    def create_mask_mge(tensor, paddings):
        shape = tensor.shape  # N,c, H,W
        inner_width = shape[2] - (paddings[0][0] + paddings[0][1])
        inner_height = shape[3] - (paddings[1][0] + paddings[1][1])
        inner_mge = F.ones([shape[0], shape[1], inner_width, inner_height])  # .float()  # need check cuda?
        outer_mge = F.zeros([shape[0], shape[1], shape[2], shape[3]])  # .float()
        outer_mge[:, :, paddings[0][0]:paddings[0][0] + inner_width, paddings[1][0]:paddings[1][0] + inner_height] = inner_mge
        # if tensor.is_cuda:
        #     inner_torch = inner_torch.cuda()
        # mask2d = F.pad(inner_mge, [paddings[0][0], paddings[0][1], paddings[1][0], paddings[1][1]])  # no padding layer

        return outer_mge

    # ==== photo loss functions ====
    def _L1(diff, occ_mask=None, if_mask_=False):
        loss_diff = F.abs(diff)
        if not if_mask_:
            photo_loss = F.mean(loss_diff)
        else:
            photo_loss = F.sum(loss_diff * occ_mask) / (F.sum(occ_mask) + 1e-6)
        return photo_loss

    def _charbonnier(diff, occ_mask=None, if_mask_=False):
        loss_diff = F.pow((diff**2 + 1e-6), 0.4)
        if not if_mask_:
            photo_loss = F.mean(loss_diff)
        else:
            photo_loss = F.sum(loss_diff * occ_mask) / (F.sum(occ_mask) + 1e-6)
        return photo_loss

    def _abs_robust(diff, occ_mask=None, if_mask_=False):
        loss_diff = F.pow((F.abs(diff) + 0.01), 0.4)
        if not if_mask_:
            photo_loss = F.mean(loss_diff)
        else:
            photo_loss = F.sum(loss_diff * occ_mask) / (F.sum(occ_mask) + 1e-6)
        return photo_loss

    img1 = _ternary_transform_mge(img1)
    img2_warp = _ternary_transform_mge(img2_warp)
    dist = _hamming_distance_mge(img1, img2_warp)
    if occu_mask1 is None:
        im_shape = img1.shape
        occu_mask1 = F.ones([im_shape[0], 1, im_shape[2], im_shape[3]])  # .float()
    transform_mask = create_mask_mge(occu_mask1, [[max_distance, max_distance], [max_distance, max_distance]])
    occ_mask = occu_mask1 * transform_mask
    # ===== compute photo loss =====
    if loss_type == 'L1':
        census_loss = _L1(dist, occ_mask, if_mask_=if_mask)
    elif loss_type == 'abs_robust':
        census_loss = _abs_robust(dist, occ_mask, if_mask_=if_mask)
    elif loss_type == 'charbonnier':
        census_loss = _charbonnier(dist, occ_mask, if_mask_=if_mask)
    else:
        raise ValueError('wrong photo loss type in census loss: %s' % loss_type)
    return census_loss


def photo_loss_census(img1, img2_warp, occu_mask1=None):
    photo_loss = _photo_loss_census(img1, img2_warp, occu_mask1=occu_mask1, max_distance=3, loss_type='charbonnier')
    return photo_loss


def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy


def smooth_grad_1st(flo, image, alpha):
    img_dx, img_dy = gradient(image)
    weights_x = F.exp(-F.mean(F.abs(img_dx), 1, keepdims=True) * alpha)
    weights_y = F.exp(-F.mean(F.abs(img_dy), 1, keepdims=True) * alpha)

    dx, dy = gradient(flo)

    loss_x = weights_x * F.abs(dx) / 2.0
    loss_y = weights_y * F.abs(dy) / 2.0
    return F.mean(loss_x) / 2.0 + F.mean(loss_y) / 2.0


def get_occu_mask_bidirection(flow12, flow21, scale=0.01, bias=0.5):
    flow21_warped = flow_warp(flow21, flow12)
    flow12_diff = flow12 + flow21_warped
    mag = F.sum((flow12 * flow12), axis=1, keepdims=True) + F.sum((flow21_warped * flow21_warped), axis=1, keepdims=True)
    occ_thresh = scale * mag + bias
    occ = F.sum((flow12_diff * flow12_diff), axis=1, keepdims=True) > occ_thresh
    return occ.astype(np.float32)


class UnFlowLoss(nn.Module):
    def __init__(self, params):
        super(UnFlowLoss, self).__init__()
        self.params = params

    def photo_loss(self, img1, img2_warp, occu_mask1):
        l1_loss = self.params.loss.l1 * F.abs(img1 - img2_warp) * occu_mask1
        ssim_loss = self.params.loss.ssim * SSIM(img1 * occu_mask1, img2_warp * occu_mask1)
        return sum([l1_loss.mean(), ssim_loss.mean()])

    def smooth_loss(self, flow, img):
        loss = smooth_grad_1st(flow, img, 10)
        return sum([loss.mean()])

    def forward(self, output, target, epoch=0):
        flows_fw, flows_bw = output["flow_fw"], output["flow_bw"]

        flow_pyrs = [F.concat([flow_fw, flow_bk], 1) for flow_fw, flow_bk in zip(flows_fw, flows_bw)]
        img1, img2 = target[:, :3], target[:, 3:]

        self.pyramid_occu_mask1 = []
        self.pyramid_occu_mask2 = []

        occu_mask1 = 1 - get_occu_mask_bidirection(flow_pyrs[0][:, :2], flow_pyrs[0][:, 2:])
        occu_mask2 = 1 - get_occu_mask_bidirection(flow_pyrs[0][:, 2:], flow_pyrs[0][:, :2])

        pyramid_smooth_losses = []
        pyramid_warp_losses = []

        for i, flow in enumerate(flow_pyrs):
            b, c, h, w = flow.shape
            if i == 0:
                s = min(h, w)
            if i == 4:
                pyramid_smooth_losses.append(0)
                pyramid_warp_losses.append(0)
                continue

            img1_rsz = F.vision.interpolate(img1, (h, w))
            img2_rsz = F.vision.interpolate(img2, (h, w))

            img1_warp = flow_warp(img2_rsz, flow[:, :2])
            img2_warp = flow_warp(img1_rsz, flow[:, 2:])

            if i != 0:
                occu_mask1 = F.vision.interpolate(occu_mask1, (h, w))
                occu_mask2 = F.vision.interpolate(occu_mask2, (h, w))

            self.pyramid_occu_mask1.append(occu_mask1)
            self.pyramid_occu_mask2.append(occu_mask2)

            if epoch < 250 and not self.params.fine_tune:
                occu_mask1 = occu_mask2 = F.ones_like(occu_mask2)

            photo_loss = self.photo_loss(img1_rsz, img1_warp, occu_mask1)
            smooth_loss = self.smooth_loss(flow[:, :2] / s, img1_rsz)

            # backward warping
            photo_loss += self.photo_loss(img2_rsz, img2_warp, occu_mask2)
            smooth_loss += self.smooth_loss(flow[:, 2:] / s, img2_rsz)

            photo_loss /= 2
            smooth_loss /= 2

            pyramid_smooth_losses.append(photo_loss)
            pyramid_warp_losses.append(smooth_loss)

            del photo_loss
            del smooth_loss

        _photo_loss = sum(pyramid_smooth_losses)
        _smooth_loss = 50 * pyramid_warp_losses[0]
        return _photo_loss + _smooth_loss


def compute_losses(data, endpoints, manager):
    loss = {}

    # compute losses
    if manager.params.loss_type == "basic":
        ce_criterion = LossCrossEntropy()

        pred = endpoints["p"]
        label = data["label"]
        loss['total'] = ce_criterion(pred, label)
    elif manager.params.loss_type == "UnFlowLoss":
        unFlowLoss = UnFlowLoss(manager.params)
        loss['total'] = unFlowLoss(endpoints, data["imgs"], manager.epoch)
    else:
        raise NotImplementedError
    return loss


def compute_metrics(data, endpoints):
    # compute metrics
    gt_flow = data["gt_flow"]
    flow_fw = endpoints["flow_fw"][0]
    return flow_error_avg(flow_fw, gt_flow)
