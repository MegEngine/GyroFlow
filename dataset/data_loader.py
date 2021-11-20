# -*- coding: utf-8 -*-
# This repo is licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import logging
import os
import cv2
import glob
import bisect

import numpy as np
import megengine.functional as F

from megengine.data import DataLoader
from megengine.data.dataset import Dataset
from megengine.data.sampler import RandomSampler, SequentialSampler

from dataset.transformations import homo_to_flow, fetch_spatial_transform, fetch_input_transform

_logger = logging.getLogger(__name__)


class ConcatDataset(Dataset):
    def __init__(self, datasets) -> None:
        self.datasets = list(datasets)

    def __getitem__(self, index):
        cumsum = np.cumsum([len(d) for d in self.datasets])
        idx_dataset = bisect.bisect_right(cumsum, index)
        offset = cumsum[idx_dataset - 1] if idx_dataset > 0 else 0
        return self.datasets[idx_dataset][index - offset]

    def __len__(self):
        return sum(len(d) for d in self.datasets)


class BaseDataset(Dataset):
    def __init__(self, input_transform, spatial_transform):
        self.input_transform = input_transform
        self.spatial_transform = spatial_transform

        self.samples = self.collect_samples()

    def collect_samples(self):
        files = glob.glob("dataset/GOF_Train/sample*")
        return files

    def resize_flow(self, inputs, target_as, isRate=False):
        h, w, _ = target_as.shape
        h_, w_, _ = inputs.shape
        res = cv2.resize(inputs, (w, h), interpolation=cv2.INTER_LINEAR)
        if isRate:
            u_scale = (w / w_)
            v_scale = (h / h_)
            res[:, :, 0] *= u_scale
            res[:, :, 1] *= v_scale
        return res

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file = self.samples[idx]
        frame_path_1 = os.path.join(file, "img1.png")
        frame_path_2 = os.path.join(file, "img2.png")

        gyro_homo_path = os.path.join(file, "gyro_homo.npy")
        gyro_homo = np.load(gyro_homo_path)

        try:
            imgs = [cv2.imread(i).astype(np.float32) for i in [frame_path_1, frame_path_2]]
        except Exception as e:
            print(frame_path_1 + " " + frame_path_2)
            raise e

        # gyro_homo is the homography from img1 to img2
        gyro_filed = homo_to_flow(np.expand_dims(gyro_homo, 0), H=600, W=800).squeeze()

        if self.spatial_transform is not None:
            imgs.append(gyro_filed)
            data = self.spatial_transform(imgs)
            imgs, gyro_filed = data[:2], data[-1]
            gyro_filed = gyro_filed.transpose(2, 0, 1)
        else:
            dummy_data = np.zeros([512, 640, 2])
            imgs = [cv2.resize(i, (640, 512)) for i in imgs]
            gyro_filed = self.resize_flow(gyro_filed, dummy_data, True).transpose(2, 0, 1)

        if self.input_transform:
            imgs_it = [self.input_transform.apply(i) for i in imgs]

        ret = {"img{}".format(i + 1): v for i, v in enumerate(imgs_it)}
        ret["gyro_field"] = gyro_filed
        return ret


class TestDataset(Dataset):
    def __init__(self, benchmark_path, input_transform):
        self.input_transform = input_transform

        self.samples = np.load(benchmark_path, allow_pickle=True)

    def __len__(self):
        return len(self.samples)

    def resize_flow(self, inputs, target_as, isRate=False):
        h, w, _ = target_as.shape
        h_, w_, _ = inputs.shape
        res = cv2.resize(inputs, (w, h), interpolation=cv2.INTER_LINEAR)
        if isRate:
            u_scale = (w / w_)
            v_scale = (h / h_)
            res[:, :, 0] *= u_scale
            res[:, :, 1] *= v_scale
        return res

    def __getitem__(self, idx):
        dummy_data = np.zeros([512, 640, 2])

        imgs = [self.samples[idx]["img1"], self.samples[idx]["img2"]]

        gyro_homo = self.samples[idx]["homo"]

        gt_flow = self.samples[idx]["gt_flow"]

        split = self.samples[idx]["split"]

        gyro_filed = homo_to_flow(np.expand_dims(gyro_homo, 0), H=600, W=800).squeeze()

        imgs = [cv2.resize(i, (640, 512)) for i in imgs]

        gt_flow = self.resize_flow(gt_flow, dummy_data, True).transpose(2, 0, 1)
        gyro_filed = self.resize_flow(gyro_filed, dummy_data, True).transpose(2, 0, 1)

        if self.input_transform:
            imgs_it = [F.transpose(i, (2, 0, 1)) for i in imgs]

        ret = {"img{}".format(i + 1): v for i, v in enumerate(imgs_it)}

        ret["gyro_field"] = gyro_filed
        ret["gt_flow"] = gt_flow
        ret["label"] = split
        ret["rain_label"] = split
        return ret


def fetch_dataloader(params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        status_manager: (class) status_manager

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    input_transform = fetch_input_transform()
    spatial_transform = fetch_spatial_transform(params)

    benchmark_path_gof_clean = "dataset/GOF_Clean.npy"
    benchmark_path_gof_final = "dataset/GOF_Final.npy"

    if params.dataset_type == "GOF":
        train_ds = BaseDataset(input_transform, spatial_transform)
        val_ds = TestDataset(benchmark_path_gof_clean, input_transform)
        test_ds = ConcatDataset(
            [TestDataset(benchmark_path_gof_clean, input_transform),
             TestDataset(benchmark_path_gof_final, input_transform)])

    dataloaders = {}
    # add defalt train data loader
    train_sampler = RandomSampler(train_ds, batch_size=params.train_batch_size, drop_last=True)
    train_dl = DataLoader(train_ds, train_sampler, num_workers=params.num_workers)
    dataloaders["train"] = train_dl

    # chosse val or test data loader for evaluate
    for split in ["val", "test"]:
        if split in params.eval_type:
            if split == "val":
                val_sampler = SequentialSampler(val_ds, batch_size=params.eval_batch_size)
                dl = DataLoader(val_ds, val_sampler, num_workers=params.num_workers)
            elif split == "test":
                test_sampler = SequentialSampler(test_ds, batch_size=params.eval_batch_size)
                dl = DataLoader(test_ds, test_sampler, num_workers=params.num_workers)
            else:
                raise ValueError("Unknown eval_type in params, should in [val, test]")
            dataloaders[split] = dl
        else:
            dataloaders[split] = None

    return dataloaders
