# Copyright 2025, the PortPy Authors
#
# Licensed under the Apache License, Version 2.0 with the Commons Clause restriction.
# You may obtain a copy of the Apache 2 License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ----------------------------------------------------------------------
# Commons Clause Restriction Notice:
# PortPy is licensed under Apache 2.0 with the Commons Clause.
# You may use, modify, and share the code for non-commercial
# academic and research purposes only.
# Commercial use — including offering PortPy as a service,
# or incorporating it into a commercial product — requires
# a separate commercial license.
# ----------------------------------------------------------------------

"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
import time
from portpy.ai.options.train_options import TrainOptions
from portpy.ai.data import create_dataset
from portpy.ai.models import create_model
from portpy.ai.util.visualizer import Visualizer
import matplotlib.pyplot as plt

import pandas as pd
import torch
import numpy as np
import copy
import sys

class DualOutput:
    def __init__(self, original_stdout, log_file):
        self.original_stdout = original_stdout
        self.log_file = log_file
    def write(self, message):
        self.original_stdout.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    def flush(self):
        self.original_stdout.flush()
        self.log_file.flush()

def main(opt):
    torch.cuda.empty_cache()
    run_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(run_dir, exist_ok=True)

    console_log_path = os.path.join(run_dir, "console_output.txt")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = open(console_log_path, 'a')
    # Redirect stdout and stderr to DualOutput
    dual_output = DualOutput(original_stdout, log_file)
    sys.stdout = dual_output
    sys.stderr = dual_output
    try:
        print(f"Saving console output to: {console_log_path}")
        # opt = TrainOptions().parse()  # get training options
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        dataset_size = len(dataset)  # get the number of images in the dataset.
        print('The number of training images = %d' % dataset_size)
        print('shuffle :{}'.format(not opt.serial_batches))
        model = create_model(opt)  # create a model given opt.model and other options
        # model.print_networks(verbose=True)
        model.setup(opt)  # regular setup: load and print networks; create schedulers
        visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
        total_iters = 0  # the total number of training iterations
        total_time_start = time.time()
        loss_summary = []
        loss_summary_test = []
        mae_summary_test = []
        r2_summary_test = []
        corr_summary_test = []
        nrmse_summary_test = []
        best_val_mae = float('inf')
        best_val_mae_epoch = -1

        # ##Create test dataset for evaluation of test loss
        optVal = copy.deepcopy(opt)
        # optVal.phase = 'test'
        optVal.phase = getattr(opt, 'val_phase', 'val')
        optVal.num_threads = 0  # test code only supports num_threads = 1
        optVal.batch_size = 1  # test code only supports batch_size = 1
        optVal.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        optVal.no_flip = True  # no flip; comment this line if results on flipped images are needed.
        optVal.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
        optVal.isTrain = False
        optVal.mode = 'eval'
        dataset_val = create_dataset(optVal)
        dataset_size_test = len(dataset_val)  # get the number of images in the dataset.
        print('The number of validation images = %d' % dataset_size_test)

        for epoch in range(opt.epoch_count,
                           opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()  # timer for data loading per iteration
            epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
            visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch

            model.set_epoch(epoch)

            total_loss_each_epoch = 0
            for i, data in enumerate(dataset):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                model.set_input(data)  # unpack data from dataset and apply preprocessing
                # if i == 0:
                #     model.save_weights(opt)
                model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

                if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                losses = model.get_current_losses()
                total_loss_each_epoch = total_loss_each_epoch + sum(losses.values())
                if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    # if opt.display_id > 0:
                    #     visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)

                iter_data_time = time.time()
            if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(epoch)

            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
            avg_loss_each_epoch_train = total_loss_each_epoch / len(dataset)
            print('Avg Loss at the end of epoch {}:{}'.format(epoch, avg_loss_each_epoch_train))
            loss_summary.append(avg_loss_each_epoch_train)

            # Evaluating Validation Loss
            total_loss_each_epoch_val = 0
            total_mae_each_epoch_val = 0
            total_r2_each_epoch_val = 0
            total_corr_each_epoch_val = 0
            total_nrmse_each_epoch_val = 0
            model.eval()
            for i, data_test in enumerate(dataset_val):
                # if i >= opt.num_test:  # only apply our model to opt.num_test images.
                #   break
                with torch.no_grad():
                    model.set_input(data_test)  # unpack data from data loader
                    model.test()
                    model.calculate_validation_loss()
                    val_losses = model.get_current_losses()
                    val_mae = model.calculate_validation_mae().item()
                    if hasattr(model, "calculate_validation_metrics"):
                        metrics = model.calculate_validation_metrics()
                        val_r2 = metrics["r2"].item()
                        val_corr = metrics["corr"].item()
                        val_nrmse = metrics["nrmse"].item()
                    else:
                        val_r2 = val_corr = val_nrmse = float("nan")
                    # print('Validation loss for iter {} is {}'.format(i+1, val_losses))
                total_loss_each_epoch_val = total_loss_each_epoch_val + sum(val_losses.values())
                total_mae_each_epoch_val += val_mae
                total_r2_each_epoch_val += val_r2
                total_corr_each_epoch_val += val_corr
                total_nrmse_each_epoch_val += val_nrmse
            avg_loss_each_epoch_val = total_loss_each_epoch_val / len(dataset_val)
            avg_mae_each_epoch_val = total_mae_each_epoch_val / len(dataset_val)
            avg_r2_each_epoch_val = total_r2_each_epoch_val / len(dataset_val)
            avg_corr_each_epoch_val = total_corr_each_epoch_val / len(dataset_val)
            avg_nrmse_each_epoch_val = total_nrmse_each_epoch_val / len(dataset_val)
            print('Avg Val Loss at the end of epoch {}:{}'.format(epoch, avg_loss_each_epoch_val))
            print('Avg Val MAE  at the end of epoch {}:{}'.format(epoch, avg_mae_each_epoch_val))
            print('Avg Val R2   at the end of epoch {}:{}'.format(epoch, avg_r2_each_epoch_val))
            print('Avg Val Corr at the end of epoch {}:{}'.format(epoch, avg_corr_each_epoch_val))
            print('Avg Val nRMSE at the end of epoch {}:{}'.format(epoch, avg_nrmse_each_epoch_val))
            loss_summary_test.append(avg_loss_each_epoch_val)
            mae_summary_test.append(avg_mae_each_epoch_val)
            r2_summary_test.append(avg_r2_each_epoch_val)
            corr_summary_test.append(avg_corr_each_epoch_val)
            nrmse_summary_test.append(avg_nrmse_each_epoch_val)
            model.metric = avg_mae_each_epoch_val
            model.update_learning_rate()  # update learning rates at the end of every epoch.
            if avg_mae_each_epoch_val < best_val_mae:
                best_val_mae = avg_mae_each_epoch_val
                best_val_mae_epoch = epoch
                print(f'New best validation MAE at epoch {epoch}: {best_val_mae:.6f}')
                model.save_networks('best_mae')
            ##Reset to training mode
            model.train()

        # plt.plot(losssummary, np.arange(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay, 1), 'r--')
        total_time_end = time.time()
        total_time = total_time_end - total_time_start
        print('Total Time for training is {}'.format(total_time))

        ##print total loss after each epoch
        loss_dir = os.path.join(opt.checkpoints_dir, opt.name)
        loss_filename = 'loss_log.txt'
        loss_filename = os.path.join(loss_dir, loss_filename)
        with open(loss_filename, 'a') as f:
            print('Total Time for training is {}'.format(total_time), file=f)

        loss_filename = 'total_loss_train.txt'
        loss_filename = os.path.join(loss_dir, loss_filename)
        # print(loss_summary)
        print(loss_filename)
        with open(loss_filename, 'w') as f:
            for i, line in enumerate(loss_summary):
                print("{} {}".format(i + 1, line), file=f)
        df = pd.read_csv(loss_filename, delim_whitespace=True, header=None)
        df.columns = ["Epoch", "Train_Loss"]
        df.to_excel(r"{}.xlsx".format(loss_filename))

        loss_filename = 'total_loss_val.txt'
        loss_filename = os.path.join(loss_dir, loss_filename)
        # print(loss_summary)
        print(loss_filename)
        with open(loss_filename, 'w') as f:
            for i, line in enumerate(loss_summary_test):
                print("{} {}".format(i + 1, line), file=f)
        df = pd.read_csv(loss_filename, delim_whitespace=True, header=None)
        df.columns = ["Epoch", "Val_Loss"]
        df.to_excel(r"{}.xlsx".format(loss_filename))

        loss_filename = 'total_mae_val.txt'
        loss_filename = os.path.join(loss_dir, loss_filename)
        print(loss_filename)
        with open(loss_filename, 'w') as f:
            for i, line in enumerate(mae_summary_test):
                print("{} {}".format(i + 1, line), file=f)

        df = pd.read_csv(loss_filename, delim_whitespace=True, header=None)
        df.columns = ["Epoch", "Val_MAE"]
        df.to_excel(r"{}.xlsx".format(loss_filename))

        for metric_name, metric_values in [
            ("total_r2_val.txt", r2_summary_test),
            ("total_corr_val.txt", corr_summary_test),
            ("total_nrmse_val.txt", nrmse_summary_test),
        ]:
            metric_path = os.path.join(loss_dir, metric_name)
            print(metric_path)
            with open(metric_path, "w") as f:
                for i, line in enumerate(metric_values):
                    print("{} {}".format(i + 1, line), file=f)

        with open(os.path.join(loss_dir, 'loss_log.txt'), 'a') as f:
            print(f'Best validation MAE: {best_val_mae} at epoch {best_val_mae_epoch}', file=f)
    finally:
        # Restore stdout and stderr, and close log file
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


def train(args=None):
    if args is None:
        # Parse command-line arguments normally
        opt = TrainOptions().parse()
    else:
        # Convert dictionary to Namespace while keeping default values
        default_opt = TrainOptions().parse()  # Get argparse parser

        # Update only the provided keys
        vars(default_opt).update(args)

        opt = default_opt  # Now opt contains updated values
        for k, v in vars(opt).items():
            print(f"{k}: {v}")
    main(opt)


if __name__ == '__main__':
    train()
