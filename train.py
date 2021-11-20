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
import imageio
import argparse
import datetime

import numpy as np
import megengine as mge
import megengine.functional as F
import megengine.distributed as dist

from tqdm import tqdm
from easydict import EasyDict
from megengine.autodiff import GradManager
from megengine.optimizer import Adam, MultiStepLR, LRScheduler

import model.net as net
import dataset.data_loader as data_loader

from common import utils
from evaluate import evaluate
from common.manager import Manager
from loss.losses import compute_losses
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file',
                    default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('-ow', '--only_weights', action='store_true', help='Only use weights to load or load all train status.')


def make_gif(img1, img2, img2_gyro_warp, img2_warp, idx=0):
    img1 = img1.numpy().transpose(1, 2, 0)
    img2 = img2.numpy().transpose(1, 2, 0)
    img2_warp = img2_warp.numpy().transpose(1, 2, 0)
    img2_gyro_warp = img2_gyro_warp.numpy().transpose(1, 2, 0)
    with imageio.get_writer('gifs/{}_test.gif'.format(idx), mode='I', duration=0.5) as writer:
        tmp_1 = np.concatenate((img1, img1, img1), 1)
        tmp_2 = np.concatenate((img2, img2_gyro_warp, img2_warp), 1)
        writer.append_data(tmp_1[:, :, ::-1])
        writer.append_data(tmp_2[:, :, ::-1])


def train(model, manager, gm, info=False):
    rank = dist.get_rank()

    # loss status and val/test status initial
    manager.reset_loss_status()

    # set model to training mode
    model.train()

    # Use tqdm for progress bar
    if rank == 0:
        t = tqdm(total=len(manager.dataloaders['train']))

    for i, data_batch in enumerate(manager.dataloaders['train']):
        # move to GPU if available
        data_batch = utils.tensor_mge(data_batch)
        data_batch["imgs"] = F.concat([data_batch["img1"], data_batch["img2"]], 1)

        # infor print
        print_str = manager.print_train_info()

        with gm:
            # compute model output and loss
            output_batch = model(data_batch)
            loss = compute_losses(data_batch, output_batch, manager)

            # update loss status and print current loss and average loss
            manager.update_loss_status(loss=loss, split="train")
            gm.backward(loss['total'])

        # performs updates using calculated gradients
        manager.optimizer.step().clear_grad()

        manager.update_step()
        if rank == 0:
            manager.writer.add_scalar("Loss/train", manager.loss_status['total'].val, manager.step)
            t.set_description(desc=print_str)
            t.update()

    if rank == 0:
        t.close()

    manager.scheduler.step()

    manager.update_epoch()


def train_and_evaluate(model, manager):
    rank = dist.get_rank()

    # reload weights from restore_file if specified
    if args.restore_file is not None:
        manager.load_checkpoints()

    world_size = dist.get_world_size()
    if world_size > 1:
        dist.bcast_list_(model.parameters())
        dist.bcast_list_(model.buffers())

    gm = GradManager().attach(
        model.parameters(),
        callbacks=dist.make_allreduce_cb("SUM") if world_size > 1 else None,
    )

    for epoch in range(manager.params.num_epochs):
        # compute number of batches in one epoch (one full pass over the training set)
        train(model, manager, gm)

        # Evaluate for one epoch on validation set
        evaluate(model, manager)

        # Save best model weights accroding to the params.major_metric
        if rank == 0:
            manager.check_best_save_last_checkpoints(latest_freq=5)


def main(params):
    mge.dtr.eviction_threshold = "5GB"
    mge.dtr.enable()

    rank = dist.get_rank()

    # Set the logger
    logger = utils.set_logger(os.path.join(params.model_dir, 'train.log'))

    # Set the tensorboard writer
    log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_dir = os.path.join(params.model_dir, "summary")
    os.makedirs(tb_dir, exist_ok=True)
    writter = SummaryWriter(log_dir=tb_dir)

    # Create the input data pipeline
    if rank == 0:
        logger.info("Loading the datasets from {}".format(params.data_dir))

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(params)

    # Define the model and optimizer
    model = net.fetch_net(params)

    # add regulizer to weights and bias
    param_groups = [
        {
            "params": utils.bias_parameters(model)
        },
        {
            "params": utils.weight_parameters(model),
            "weight_decay": 1e-6
        },
    ]

    optimizer = Adam(param_groups, lr=params.learning_rate, eps=1e-7)
    milestones = [50, 150, 250, 350, 450]
    scheduler = MultiStepLR(optimizer, milestones, 0.5)

    # initial status for checkpoint manager
    manager = Manager(model=model,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      params=params,
                      dataloaders=dataloaders,
                      writer=writter,
                      logger=logger)

    # Train the model
    if rank == 0:
        logger.info("Starting training for {} epoch(s)".format(params.num_epochs))

    train_and_evaluate(model, manager)


if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)

    with open(json_path) as f:
        params = EasyDict(json.load(f))
    params.update(vars(args))

    train_proc = dist.launcher(main) if mge.device.get_device_count("gpu") > 1 else main
    train_proc(params)
