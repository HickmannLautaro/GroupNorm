import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.notebook import tqdm, trange

from ResNet_network import load_model_weights
from train_utils import get_test_data


def get_mean_std(norm, batch):
    dat = np.array([pd.read_csv(f"train_outputs/logs/{norm}/ResNet20/no_weight_decay/epochs_30/batch_{batch}/run_{run}/training.log")['val_sparse_categorical_accuracy'].values for run in ["1", "2", "3", "4", "5"]])
    dat = 100 - dat * 100
    mean = dat.mean(axis=0)
    std_val = dat.std(axis=0)
    return mean, std_val


def get_error_lines(x, norm):
    for batch in ["2", "4", "8", "16", "32"]:
        mean, std_val = get_mean_std(norm, batch)
        plt.errorbar(x, mean, yerr=std_val, label=f"{norm}, batch {batch}", capsize=2)


def plot_validation_comparisson():
    fig = plt.figure(figsize=(20, 8))
    x = np.arange(1, 31)
    plt.suptitle("Validation classification error on the test set (mean of 5 runs with std)")

    plt.subplot(1, 2, 1)
    plt.title("Batch Norm (BN)")
    plt.xlabel("Training epochs")
    plt.ylabel("classification error (%)")
    get_error_lines(x, 'BN')
    plt.legend()
    plt.gca().yaxis.grid(True)
    plt.axvline(8 + 1, color='gray', alpha=0.35)
    plt.axvline(15 + 1, color='gray', alpha=0.35)
    plt.axvline(22 + 1, color='gray', alpha=0.35)

    plt.subplot(1, 2, 2)
    plt.title("Group Norm (GN)")

    get_error_lines(x, 'GN')
    plt.xlabel("Training epochs")
    plt.ylabel("classification error (%)")
    plt.legend()
    plt.gca().yaxis.grid(True)
    plt.axvline(8 + 1, color='gray', alpha=0.35)
    plt.axvline(15 + 1, color='gray', alpha=0.35)
    plt.axvline(22 + 1, color='gray', alpha=0.35)


def get_median_std(norm, batch):
    dat = np.array([pd.read_csv(f"train_outputs/logs/{norm}/ResNet20/no_weight_decay/epochs_30/batch_{batch}/run_{run}/training.log")['val_sparse_categorical_accuracy'].values for run in ["1", "2", "3", "4", "5"]])
    dat = 100 - dat * 100
    dat = dat[:, -1]
    median = np.median(dat, axis=0)
    std_val = dat.std(axis=0)
    return median, std_val


def get_error_points(x, norm):
    medians = []
    std_vals = []
    for batch in ["32", "16", "8", "4", "2"]:
        median, std_val = get_median_std(norm, batch)
        medians.append(median)
        std_vals.append(std_val)
    plt.errorbar(x, medians, yerr=std_val, label=norm, capsize=2)


def GN_vs_BN_last_epoch():
    fig = plt.figure(figsize=(20, 8))

    x = range(5)
    plt.title("Batch Norm (BN) vs Group Norm (GN) (median of the last epoch of 5 models with standard deviation)")

    get_error_points(x, "BN")
    get_error_points(x, "GN")

    plt.xlabel("Batch size")
    plt.ylabel("classification error (%)")
    plt.gca().yaxis.grid(True)
    plt.xticks(x, ["32", "16", "8", "4", "2"])
    plt.legend()


def get_curve_from_models(norm, only_new):
    medians = []
    std_vals = []
    for batch in tqdm([32, 16, 8, 4, 2], desc="Batch", leave=False):
        arguments = {
            'batch_size': batch,
            'norm': norm,
            'weight_decay': False,
            'experimental_data_aug': False,
            'ResNet': 3,
            'epochs': 30}
        test_data = get_test_data(arguments)
        evaluation = []
        for run in trange(1, 6, desc="run", leave=False):
            if not only_new:
                old_struc = True
                if arguments['norm'] == 'BN':
                    if batch == 32 and run in [2, 3, 4, 5]:
                        old_struc = False
                    elif batch == 16 and run in [4, 5]:
                        old_struc = False
                elif arguments['norm'] == 'GN' and batch == 8 and run in [2]:
                    old_struc = False
            else:
                old_struc = False

            arguments['run'] = run
            model_path = os.path.join("./train_outputs/checkpoints", arguments['norm'], "ResNet" + str(arguments['ResNet'] * 6 + 2), "no_weight_decay", "epochs_" + str(arguments['epochs']), "batch_" + str(arguments['batch_size']), "run_" + str(arguments['run']))
            model = load_model_weights(arguments, os.path.join(model_path, os.listdir(model_path)[0]), old_struc=old_struc)
            evaluation.append(model.evaluate(test_data, verbose=0)[-1])
            tf.keras.backend.clear_session()
        evaluation = np.array(evaluation)
        dat = 100 - evaluation * 100
        medians.append(np.median(dat, axis=0))
        std_vals.append(dat.std(axis=0))
    return medians, std_vals


def GN_vs_BN_best_epoch(only_new):
    fig = plt.figure(figsize=(20, 8))

    x = range(5)
    plt.title("Batch Norm (BN) vs Group Norm (GN) (median of the best epoch of 5 models with standard deviation)")

    for norm in tqdm(["BN", "GN"], desc="Normalization"):
        medians, std_vals = get_curve_from_models(norm, only_new)
        plt.errorbar(x, medians, yerr=std_vals, label=norm, capsize=2)

    plt.xlabel("Batch size")
    plt.ylabel("classification error (%)")
    plt.gca().yaxis.grid(True)
    plt.xticks(x, ["32", "16", "8", "4", "2"])
    plt.legend()
