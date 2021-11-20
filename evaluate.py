# -*- coding: utf-8 -*-
# This repo is licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import os
import json
import logging
import argparse

import megengine.distributed as dist
import megengine.functional as F

import model.net as net
import dataset.data_loader as data_loader

from easydict import EasyDict

from common import utils
from common.manager import Manager
from loss.losses import compute_losses, compute_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir containing weights to load")


def evaluate(model, manager):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # set model to evaluation mode
    model.eval()

    # compute metrics over the dataset
    if manager.dataloaders["val"] is not None:
        # loss status and val status initial
        manager.reset_loss_status()
        manager.reset_metric_status("val")
        for data_batch in manager.dataloaders["val"]:
            # compute the real batch size
            bs = data_batch["img1"].shape[0]
            # move to GPU if available
            data_batch = utils.tensor_mge(data_batch)

            data_batch["imgs"] = F.concat([data_batch["img1"] / 255.0, data_batch["img2"] / 255.0], 1)
            # compute model output
            output_batch = model(data_batch)
            # compute all loss on this batch
            # loss = compute_losses(data_batch, output_batch, manager.params)
            metrics = {}
            metrics["EPE"] = compute_metrics(data_batch, output_batch)
            if world_size > 1:
                # loss['total'] = F.distributed.all_reduce_sum(loss['total']) / world_size
                metrics['EPE'] = F.distributed.all_reduce_sum(metrics['EPE']) / world_size
            # manager.update_loss_status(loss, "val", bs)
            # compute all metrics on this batch

            manager.update_metric_status(metrics, "val", bs)
            # manager.print_metrics("val", title="Val", color="green")

        # update data to tensorboard
        if rank == 0:
            # manager.writer.add_scalar("Loss/val", manager.loss_status["total"].avg, manager.epoch)
            # manager.logger.info("Loss/valid epoch {}: {}".format(manager.epoch, manager.loss_status['total'].avg))

            for k, v in manager.val_status.items():
                manager.writer.add_scalar("Metric/val/{}".format(k), v.avg, manager.epoch)
                # manager.logger.info("Metric/valid epoch {}: {}".format(manager.epoch, v.avg))
            # For each epoch, print the metric
            manager.print_metrics("val", title="Val", color="green")


def test(model, manager):
    # set model to evaluation mode
    model.eval()

    if manager.dataloaders["test"] is not None:
        # loss status and test status initial
        manager.reset_loss_status()
        manager.reset_metric_status("test")
        for data_batch in manager.dataloaders["test"]:
            # compute the real batch size
            bs = data_batch["img1"].shape[0]
            # move to GPU if available
            data_batch = utils.tensor_mge(data_batch)
            data_batch["imgs"] = F.concat([data_batch["img1"], data_batch["img2"]], 1)
            # compute model output
            output_batch = model(data_batch)
            # compute all metrics on this batch
            metrics = {}

            # identity_batch = {"flow_fw": [F.zeros_like(data_batch["gyro_field"])]}
            # metrics["I33"] = compute_metrics(data_batch, identity_batch)

            # gyro_batch = {"flow_fw": [data_batch["gyro_field"]]}
            # metrics["GyroField"] = compute_metrics(data_batch, gyro_batch)

            metrics["EPE"] = compute_metrics(data_batch, output_batch)

            if data_batch["label"][0] == "RE":
                metrics["RE"] = compute_metrics(data_batch, output_batch)
            elif data_batch["label"][0] == "Rain":
                metrics["Rain"] = compute_metrics(data_batch, output_batch)
            elif data_batch["label"][0] == "Dark":
                metrics["Dark"] = compute_metrics(data_batch, output_batch)
            elif data_batch["label"][0] == "Fog":
                metrics["Fog"] = compute_metrics(data_batch, output_batch)

            manager.update_metric_status(metrics, "test", bs)

            manager.print_metrics("test", title="Test", color="red")

        # For each epoch, print the metric
        print("The average results are: ")
        manager.print_metrics("test", title="Test", color="red")


if __name__ == '__main__':
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    with open(json_path) as f:
        params = EasyDict(json.load(f))
    # Only load model weights
    params.only_weights = True

    # Update args into params
    params.update(vars(args))

    # Get the logger
    logger = utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # Fetch dataloaders
    params.eval_type = 'test'
    dataloaders = data_loader.fetch_dataloader(params)

    # Define the model and optimizer
    model = net.fetch_net(params)

    # Initial status for checkpoint manager
    manager = Manager(model=model, optimizer=None, scheduler=None, params=params, dataloaders=dataloaders, writer=None, logger=logger)

    # Reload weights from the saved file
    manager.load_checkpoints()

    # Test the model
    logger.info("Starting test")

    # Evaluate
    test(model, manager)
