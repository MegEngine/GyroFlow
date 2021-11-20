# -*- coding: utf-8 -*-
# This repo is licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import json
import logging
import megengine as mge
import coloredlogs

import numpy as np
import megengine.functional as F


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, dict):
        """Loads parameters from json file"""
        self.__dict__.update(dict)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.val_previous = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val_previous = self.val
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def loss_meter_manager_intial(loss_meter_names):
    loss_meters = []
    for name in loss_meter_names:
        exec("%s = %s" % (name, 'AverageMeter()'))
        exec("loss_meters.append(%s)" % name)

    return loss_meters


def tensor_mge(batch, check_on=True):
    if check_on:
        for k, v in batch.items():
            if isinstance(v, np.ndarray):
                batch[k] = mge.Tensor(v)
    else:
        for k, v in batch.items():
            batch[k] = v.numpy()
    return batch


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s %(name)s %(message)s')
    file_handler = logging.FileHandler(log_path)
    log_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.info('Output and logs will be saved to {}'.format(log_path))
    return logger


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    save_dict = {}
    with open(json_path, "w") as f:
        # We need to convert the values to float for json (it doesn"t accept np.array, np.float, )
        for k, v in d.items():
            if isinstance(v, AverageMeter):
                save_dict[k] = float(v.avg)
            else:
                save_dict[k] = float(v)
        json.dump(save_dict, f, indent=4)


def upsample2d_flow_as(inputs, target_as, mode="bilinear", if_rate=False):
    _, _, h, w = target_as.shape
    res = F.vision.interpolate(inputs, [h, w], mode=mode, align_corners=True)
    _, _, h_, w_ = inputs.shape
    if if_rate:
        u_scale = (w / w_)
        v_scale = (h / h_)
        res[:, 0] *= u_scale
        res[:, 1] *= v_scale
    return res


def mesh_grid(B, H, W):
    # mesh grid
    x_base = F.arange(0, W)
    x_base = F.tile(x_base, (B, H, 1))

    y_base = F.arange(0, H)  # BHW
    y_base = F.tile(y_base, (B, W, 1)).transpose(0, 2, 1)

    base_grid = F.stack([x_base, y_base], 1)  # B2HW
    return base_grid


def flow_warp(x, flow12):
    B, _, H, W = x.shape

    base_grid = mesh_grid(B, H, W).astype(x)  # B2HW

    grid_warp = base_grid + flow12
    grid_warp = F.transpose(grid_warp, (0, 2, 3, 1))

    warp_imgs = F.vision.remap(x, grid_warp)
    return warp_imgs


def euclidean(t):
    return F.sqrt(F.sum(t**2, axis=(1, ), keepdims=True))


def flow_error_avg(pred_flow, gt_flow):
    _, _, H, W = gt_flow.shape
    _, _, h, w = pred_flow.shape
    assert (H == h) and (W == w), "inps shape is not the same: {} - {}".format((H, W), (h, w))

    diff = euclidean(pred_flow - gt_flow)
    diff_s = F.mean(diff)
    error = diff_s
    return error


def weight_parameters(module):
    return [param for name, param in module.named_parameters() if "weight" in name]


def bias_parameters(module):
    return [param for name, param in module.named_parameters() if "bias" in name]
