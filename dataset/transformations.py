# -*- coding: utf-8 -*-
# This repo is licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import random
import numbers

import numpy as np
import megengine as mge
import megengine.functional as F
import megengine.data.transform as T


class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs):
        h, w, _ = inputs[0].shape
        th, tw = self.size
        if w == tw and h == th:
            return inputs

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        inputs = [img[y1:y1 + th, x1:x1 + tw] for img in inputs]
        return inputs


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5"""
    def __call__(self, inputs):
        if random.random() < 0.5:
            inputs = [np.copy(np.fliplr(im)) for im in inputs]
        return inputs


class RandomSwap(object):
    def __call__(self, inputs):
        if random.random() < 0.5:
            inputs = inputs[::-1]
        return inputs


class Compose(object):
    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input):
        for t in self.co_transforms:
            input = t(input)
        return input


def mesh_grid_mge(B, H, W):
    # mesh grid
    x_base = F.arange(0, W)
    x_base = F.tile(x_base, (B, H, 1))

    y_base = F.arange(0, H)  # BHW
    y_base = F.tile(y_base, (B, W, 1)).transpose(0, 2, 1)

    ones = F.ones_like(x_base)

    base_grid = F.stack([x_base, y_base, ones], 1)  # B3HW
    return base_grid


def get_flow_mge(H_mat_mul, patch_indices, image_size_h=600, image_size_w=800):
    # (N, 6, 3, 3)
    batch_size = H_mat_mul.shape[0]
    divide = H_mat_mul.shape[1]
    H_mat_mul = mge.Tensor(H_mat_mul.reshape(batch_size, divide, 3, 3))

    small_patch_sz = [image_size_h // divide, image_size_w]
    small = 1e-7

    H_mat_pool = F.zeros((batch_size, image_size_h, image_size_w, 3, 3))

    for i in range(divide):
        H_mat = H_mat_mul[:, i, :, :]

        if i == divide - 1:
            H_mat = F.broadcast_to(F.expand_dims(F.expand_dims(H_mat, 1), 1),
                                   (batch_size, image_size_h - i * small_patch_sz[0], image_size_w, 3, 3))
            H_mat_pool[:, i * small_patch_sz[0]:, ...] = H_mat
            continue

        H_mat = F.broadcast_to(F.expand_dims(F.expand_dims(H_mat, 1), 1), (batch_size, small_patch_sz[0], image_size_w, 3, 3))
        H_mat_pool[:, i * small_patch_sz[0]:(i + 1) * small_patch_sz[0], ...] = H_mat

    pred_I2_index_warp = F.expand_dims(patch_indices.transpose(0, 2, 3, 1), 4)
    pred_I2_index_warp = F.matmul(H_mat_pool, pred_I2_index_warp)[:, :, :, :, 0].transpose(0, 3, 1, 2)
    T_t = pred_I2_index_warp[:, 2:3, ...]
    smallers = 1e-6
    T_t = T_t + smallers
    v1 = pred_I2_index_warp[:, 0:1, ...]
    v2 = pred_I2_index_warp[:, 1:2, ...]
    v1 = v1 / T_t
    v2 = v2 / T_t
    warp_index = F.concat((v1, v2), 1)
    vgrid = patch_indices[:, :2, ...]

    flow = warp_index - vgrid
    return flow


def homo_to_flow_mge(homo, H=600, W=800):
    img_indices = mesh_grid_mge(B=1, H=H, W=W)
    flow_gyro = get_flow_mge(homo, img_indices, image_size_h=H, image_size_w=W)
    return flow_gyro


def mesh_grid(B, H, W):
    # mesh grid
    x_base = np.arange(0, W)
    x_base = np.tile(x_base, (B, H, 1))

    y_base = np.arange(0, H)  # BHW
    y_base = np.tile(y_base, (B, W, 1)).transpose(0, 2, 1)

    ones = np.ones_like(x_base)

    base_grid = np.stack([x_base, y_base, ones], 1)  # B3HW
    return base_grid


def get_flow(H_mat_mul, patch_indices, image_size_h=600, image_size_w=800):
    # (N, 6, 3, 3)
    batch_size = H_mat_mul.shape[0]
    divide = H_mat_mul.shape[1]
    H_mat_mul = H_mat_mul.reshape(batch_size, divide, 3, 3)

    small_patch_sz = [image_size_h // divide, image_size_w]
    small = 1e-7

    H_mat_pool = np.zeros((batch_size, image_size_h, image_size_w, 3, 3))

    for i in range(divide):
        H_mat = H_mat_mul[:, i, :, :]

        if i == divide - 1:
            H_mat = np.broadcast_to(np.expand_dims(np.expand_dims(H_mat, 1), 1),
                                    (batch_size, image_size_h - i * small_patch_sz[0], image_size_w, 3, 3))
            H_mat_pool[:, i * small_patch_sz[0]:, ...] = H_mat
            continue

        H_mat = np.broadcast_to(np.expand_dims(np.expand_dims(H_mat, 1), 1), (batch_size, small_patch_sz[0], image_size_w, 3, 3))
        H_mat_pool[:, i * small_patch_sz[0]:(i + 1) * small_patch_sz[0], ...] = H_mat

    pred_I2_index_warp = np.expand_dims(patch_indices.transpose(0, 2, 3, 1), 4)
    pred_I2_index_warp = np.matmul(H_mat_pool, pred_I2_index_warp)[:, :, :, :, 0].transpose(0, 3, 1, 2)
    T_t = pred_I2_index_warp[:, 2:3, ...]
    smallers = 1e-6
    T_t = T_t + smallers
    v1 = pred_I2_index_warp[:, 0:1, ...]
    v2 = pred_I2_index_warp[:, 1:2, ...]
    v1 = v1 / T_t
    v2 = v2 / T_t
    warp_index = np.concatenate((v1, v2), 1)
    vgrid = patch_indices[:, :2, ...]

    flow = warp_index - vgrid
    # NCHW to HWC
    return flow.squeeze().transpose(1, 2, 0)


def homo_to_flow(homo, H=600, W=800):
    img_indices = mesh_grid(B=1, H=H, W=W)
    flow_gyro = get_flow(homo, img_indices, image_size_h=H, image_size_w=W)
    return flow_gyro


def fetch_input_transform():
    transformer = T.Compose([T.ToMode(), T.Normalize(mean=0, std=255.0)])
    return transformer


def fetch_spatial_transform(params):
    transforms = []
    if params.data_aug.crop:
        transforms.append(RandomCrop(params.data_aug.para_crop))
    if params.data_aug.hflip:
        transforms.append(RandomHorizontalFlip())
    if params.data_aug.swap:
        transforms.append(RandomSwap())
    return Compose(transforms)
