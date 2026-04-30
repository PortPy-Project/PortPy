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

"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
# Copyright 2025, the PortPy Authors
#
# Licensed under the Apache License, Version 2.0 with the Commons Clause restriction.

import os
import sys
import numpy as np

# used when running from CLI script to find project root and append it to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from portpy.ai.options.test_options import TestOptions
from portpy.ai.data import create_dataset
from portpy.ai.models import create_model
from portpy.ai.util.visualizer import save_images
from portpy.ai.util import html

# Select the variable names of the npz images
npy_out_fnames = ["CT2DOSE"]


def _get_web_dir(opt):
    web_dir = os.path.join(
        opt.results_dir,
        opt.name,
        "{}_{}".format(opt.phase, opt.epoch)
    )
    if opt.load_iter > 0:
        web_dir = "{:s}_iter{:d}".format(web_dir, opt.load_iter)
    return web_dir


def _get_path_name(a_paths):
    if isinstance(a_paths, (list, tuple)):
        return os.path.basename(a_paths[0])
    return os.path.basename(a_paths)


def _get_path_full(a_paths):
    if isinstance(a_paths, (list, tuple)):
        return a_paths[0]
    return a_paths


def _get_beamlet_patient_name(data):
    """
    BeamletDose3DDataset returns data["meta"]["patient_id"].
    Fallback uses folder structure:
        root / phase / patient_id / beamlets / bl_beam*_col*.npz
    """
    try:
        pid = data["meta"]["patient_id"]
        if isinstance(pid, (list, tuple)):
            return str(pid[0])
        if hasattr(pid, "item"):
            return str(pid.item())
        return str(pid)
    except Exception:
        a_path = _get_path_full(data["A_paths"])
        return os.path.basename(os.path.dirname(os.path.dirname(a_path)))


def _calculate_metrics(model):
    """
    Backward-compatible metric calculation.

    For newer models, use calculate_validation_metrics().
    For older total-dose models, use calculate_validation_mae().
    """
    model.calculate_validation_loss()
    val_losses = model.get_current_losses()
    case_loss = sum(val_losses.values())

    if hasattr(model, "calculate_validation_metrics"):
        metrics = model.calculate_validation_metrics()
        val_mae = float(metrics["mae"].item())
        val_r2 = float(metrics["r2"].item())
        val_corr = float(metrics["corr"].item())
        val_nrmse = float(metrics["nrmse"].item())
    else:
        val_mae = float(model.calculate_validation_mae().item())
        val_r2 = float("nan")
        val_corr = float("nan")
        val_nrmse = float("nan")

    return {
        "loss": case_loss,
        "mae": val_mae,
        "r2": val_r2,
        "corr": val_corr,
        "nrmse": val_nrmse,
    }


def run_dose_prediction_test(opt, dataset, model):
    """
    Default test workflow for total dose prediction.
    Saves HTML/images/npz outputs and patient-level metrics.
    """
    total_test_loss = 0.0
    total_test_mae = 0.0
    n_eval = 0
    per_patient_metrics = []

    web_dir = _get_web_dir(opt)
    print("creating web directory", web_dir)
    webpage = html.HTML(
        web_dir,
        "Experiment = %s, Phase = %s, Epoch = %s"
        % (opt.name, opt.phase, opt.epoch)
    )

    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()

        # Evaluate only if GT is available
        if "B" in data and data["B"] is not None:
            try:
                metrics = _calculate_metrics(model)

                total_test_loss += metrics["loss"]
                total_test_mae += metrics["mae"]
                n_eval += 1

                patient_name = _get_path_name(data["A_paths"])
                per_patient_metrics.append({
                    "patient": patient_name,
                    "loss": metrics["loss"],
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                    "corr": metrics["corr"],
                    "nrmse": metrics["nrmse"],
                })

            except Exception as e:
                print(f"Skipping metric evaluation for this case: {e}")

        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()

        if i % 5 == 0:
            print("processing (%04d)-th image... %s" % (i, img_path))

        save_images(
            webpage,
            visuals,
            img_path,
            aspect_ratio=opt.aspect_ratio,
            width=opt.display_winsize,
            npy_out_fnames=npy_out_fnames
        )

    webpage.save()

    if n_eval > 0:
        print("Avg Test Loss: {}".format(total_test_loss / n_eval))
        print("Avg Test MAE: {}".format(total_test_mae / n_eval))

        per_patient_metrics = sorted(
            per_patient_metrics,
            key=lambda x: x["mae"],
            reverse=True
        )

        metrics_file = os.path.join(web_dir, "patient_metrics.txt")
        with open(metrics_file, "w") as f:
            for row in per_patient_metrics:
                f.write(
                    "Patient: {}, MAE: {}, R2: {}, Corr: {}, nRMSE: {}, Loss: {}\n".format(
                        row["patient"],
                        row["mae"],
                        row["r2"],
                        row["corr"],
                        row["nrmse"],
                        row["loss"],
                    )
                )
    else:
        print("No GT available. Only inference outputs were saved.")


def run_beamlet_dose_prediction_test(opt, dataset, model):
    """
    Test workflow for beamlet dose prediction.

    Does not save HTML/images for each beamlet because there can be many samples.
    Saves:
        sample_metrics.txt   -> per beamlet/sample
        patient_metrics.txt  -> averaged over beamlets for each patient
    """
    total_test_loss = 0.0
    total_test_mae = 0.0
    n_eval = 0

    per_sample_metrics = []
    patient_metric_accum = {}

    web_dir = _get_web_dir(opt)
    os.makedirs(web_dir, exist_ok=True)

    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()

        if "B" in data and data["B"] is not None:
            try:
                metrics = _calculate_metrics(model)

                total_test_loss += metrics["loss"]
                total_test_mae += metrics["mae"]
                n_eval += 1

                sample_name = _get_path_name(data["A_paths"])
                patient_name = _get_beamlet_patient_name(data)

                row = {
                    "sample": sample_name,
                    "patient": patient_name,
                    "loss": metrics["loss"],
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                    "corr": metrics["corr"],
                    "nrmse": metrics["nrmse"],
                }
                per_sample_metrics.append(row)

                if patient_name not in patient_metric_accum:
                    patient_metric_accum[patient_name] = {
                        "loss": [],
                        "mae": [],
                        "r2": [],
                        "corr": [],
                        "nrmse": [],
                    }

                for k in patient_metric_accum[patient_name]:
                    patient_metric_accum[patient_name][k].append(row[k])

            except Exception as e:
                print(f"Skipping metric evaluation for this beamlet sample: {e}")

        if i % 500 == 0:
            print("processing beamlet sample (%06d)..." % i)

    if n_eval > 0:
        print("Avg Test Loss: {}".format(total_test_loss / n_eval))
        print("Avg Test MAE: {}".format(total_test_mae / n_eval))

        # Sample-level metrics
        per_sample_metrics = sorted(
            per_sample_metrics,
            key=lambda x: x["mae"],
            reverse=True
        )

        sample_metrics_file = os.path.join(web_dir, "sample_metrics.txt")
        with open(sample_metrics_file, "w") as f:
            for row in per_sample_metrics:
                f.write(
                    "Sample: {}, Patient: {}, MAE: {}, R2: {}, Corr: {}, nRMSE: {}, Loss: {}\n".format(
                        row["sample"],
                        row["patient"],
                        row["mae"],
                        row["r2"],
                        row["corr"],
                        row["nrmse"],
                        row["loss"],
                    )
                )

        # Patient-level average across beamlets
        patient_metrics_file = os.path.join(web_dir, "patient_metrics.txt")
        with open(patient_metrics_file, "w") as f:
            for patient_name, vals in sorted(patient_metric_accum.items()):
                f.write(
                    "Patient: {}, MAE: {}, R2: {}, Corr: {}, nRMSE: {}, Loss: {}, NumSamples: {}\n".format(
                        patient_name,
                        np.nanmean(vals["mae"]),
                        np.nanmean(vals["r2"]),
                        np.nanmean(vals["corr"]),
                        np.nanmean(vals["nrmse"]),
                        np.nanmean(vals["loss"]),
                        len(vals["mae"]),
                    )
                )
    else:
        print("No GT available. No metrics were saved.")


def main(opt):
    # hard-code test parameters
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    if opt.eval:
        model.eval()

    test_task = getattr(opt, "test_task", "dose_prediction")

    if test_task == "dose_prediction":
        run_dose_prediction_test(opt, dataset, model)
    elif test_task == "beamlet_dose_prediction":
        run_beamlet_dose_prediction_test(opt, dataset, model)
    else:
        raise ValueError(f"Unsupported test_task: {test_task}")


def test(args=None):
    if args is None:
        opt = TestOptions().parse()
    else:
        default_opt = TestOptions().parse()
        vars(default_opt).update(args)
        opt = default_opt

        for k, v in vars(opt).items():
            print(f"{k}: {v}")

    main(opt)


if __name__ == "__main__":
    test()