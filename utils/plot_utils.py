import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.notebook import tqdm, trange

from ResNet_network import load_model_weights
from train_utils import get_test_data


def get_mean_std(norm, batch):
    """
    Calculates the mean and standard deviation of the val classification error over 5 runs for all epochs.
    @param norm: What normalization to load, i.e. 'GN' or 'BN'.
    @param batch: Batch size to load.
    @return: mean and std
    """
    dat = np.array([pd.read_csv(f"train_outputs/logs/{norm}/ResNet20/no_weight_decay/epochs_30/batch_{batch}/run_{run}/training.log")['val_sparse_categorical_accuracy'].values for run in ["1", "2", "3", "4", "5"]])
    dat = 100 - dat * 100  # Convert val accuracy to classification error
    mean = dat.mean(axis=0)
    std_val = dat.std(axis=0)
    return mean, std_val


def get_error_lines(x, norm):
    """
    Plots the validation lines for all epochs for all batch sizes
    @param x: Epoch range for plotting range(1,31)
    @param norm: What norm to plot, i.e. 'GN' or 'BN'
    """
    for batch in ["2", "4", "8", "16", "32"]:
        mean, std_val = get_mean_std(norm, batch)
        plt.errorbar(x, mean, yerr=std_val, label=f"{norm}, batch {batch}", capsize=2)  # Plot lines with error std as error bar


def plot_validation_comparison():
    """
    Plots the validation curves for all batch sizes in one subplot per normalization type
    """
    fig = plt.figure(figsize=(20, 8))
    x = np.arange(1, 31)  # Epoch range
    plt.suptitle("Validation classification error on the test set (mean of 5 runs with std)")

    for i, norm in enumerate(['BN', 'GN'], 1):
        plt.subplot(1, 2, i)
        plt.title(f"Batch Norm ({norm})")
        plt.xlabel("Training epochs")
        plt.ylabel("classification error (%)")
        get_error_lines(x, norm)
        plt.legend()
        plt.gca().yaxis.grid(True)  # Set grid to only horizontal lines
        # Draw vertical lines in the epochs where lr changed.
        plt.axvline(8 + 1, color='gray', alpha=0.35)
        plt.axvline(15 + 1, color='gray', alpha=0.35)
        plt.axvline(22 + 1, color='gray', alpha=0.35)


def get_median_std(norm, batch):
    """
    Calculates the median and standard deviation of the val classification error over 5 runs for the last epochs.
    @param norm: What normalization to load, i.e. 'GN' or 'BN'.
    @param batch: Batch size to load.
    @return: median and std
    """
    dat = np.array([pd.read_csv(f"train_outputs/logs/{norm}/ResNet20/no_weight_decay/epochs_30/batch_{batch}/run_{run}/training.log")['val_sparse_categorical_accuracy'].values for run in ["1", "2", "3", "4", "5"]])
    dat = 100 - dat * 100  # Convert val accuracy to classification error
    dat = dat[:, -1]  # Use only the last epoch of each run
    median = np.median(dat, axis=0)
    std_val = dat.std(axis=0)
    return median, std_val


def get_error_points(x, norm):
    """
     Plots the val classification error points as lines for all batch sizes
     @param x: Batch size range for plotting range(5)
     @param norm: What norm to plot, i.e. 'GN' or 'BN'
     """
    medians = []
    std_vals = []
    for batch in ["32", "16", "8", "4", "2"]:
        median, std_val = get_median_std(norm, batch)
        medians.append(median)
        std_vals.append(std_val)
    plt.errorbar(x, medians, yerr=std_val, label=norm, capsize=2)  # Plot lines with error std as error bar


def GN_vs_BN_last_epoch():
    """
    Plots the GN vs BN classification error in one plot over all batch sizes using the results of the validation of last epoch
    """
    fig = plt.figure(figsize=(20, 8))

    x = range(5)  # 5 batch sizes
    plt.title("Batch Norm (BN) vs Group Norm (GN) (median of the last epoch of 5 models with standard deviation)")

    get_error_points(x, "BN")
    get_error_points(x, "GN")

    plt.xlabel("Batch size")
    plt.ylabel("classification error (%)")
    plt.gca().yaxis.grid(True)
    plt.xticks(x, ["32", "16", "8", "4", "2"])  # Change x ticks labels
    plt.legend()
    plt.savefig("Images/GN_vs_BN_last_epoch.png", bbox_inches='tight')


def get_curve_from_models(norm, only_new):
    """
    Loads models from disc and evaluates them for all runs and batch sizes given
    @param norm: What norm to load and plot, i.e. 'GN' or 'BN'
    @param only_new: I made a small change in the order in how the layers are in the initialization of the resnet class and therefore small changes must be made in how these models are loaded. This can be deactivated by setting only_new = true
    @return: medians, std_vals over all batches for a normalization type
    """
    medians = []
    std_vals = []
    for batch in tqdm([32, 16, 8, 4, 2], desc="Batch", leave=False):  # Iterate over all batch sizes
        # To load the model weights a model must first be created for this the parsed input arguments must be defined
        arguments = {
            'batch_size': batch,
            'norm': norm,
            'weight_decay': False,
            'experimental_data_aug': False,
            'ResNet': 3,
            'epochs': 30}
        test_data = get_test_data(arguments)
        evaluation = []
        for run in trange(1, 6, desc="run", leave=False):  # Over all runs

            if not only_new:
                # Some specific models have the newer initialization order for these the odd_struct must be set
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
            # get model path
            model_path = os.path.join("./train_outputs/checkpoints", arguments['norm'], "ResNet" + str(arguments['ResNet'] * 6 + 2), "no_weight_decay", "epochs_" + str(arguments['epochs']), "batch_" + str(arguments['batch_size']), "run_" + str(arguments['run']))
            # Filter for .DS_STORE in macOS because it keeps creating them :(
            checkpoints = os.listdir(model_path)
            checkpoints = [x for x in checkpoints if '.DS_STORE' not in x]
            # Call custom function to create a model and load the trained weighs. Using the checkpoint saved for the best epoch.
            model = load_model_weights(arguments, os.path.join(model_path, checkpoints[0]), old_struc=old_struc)
            # Evaluate the model
            evaluation.append(model.evaluate(test_data, verbose=0)[-1])
            tf.keras.backend.clear_session()  # Free up memory
        evaluation = np.array(evaluation)
        dat = 100 - evaluation * 100  # Convert val accuracy to classification error
        medians.append(np.median(dat, axis=0))
        std_vals.append(dat.std(axis=0))
    return medians, std_vals


def GN_vs_BN_best_epoch(only_new):
    """
    Plots the GN vs BN classification error in one plot over all batch sizes loading the model weights and evaluating them.
    @param only_new: I made a small change in the order in how the layers are in the initialization of the resnet class and therefore small changes must be made in how these models are loaded. This can be deactivated by setting only_new = true
    """
    fig = plt.figure(figsize=(20, 8))

    x = range(5)  # 5 batch sizes
    plt.title("Batch Norm (BN) vs Group Norm (GN) (median of the best epoch of 5 models with standard deviation)")

    for norm in tqdm(["BN", "GN"], desc="Normalization"):
        medians, std_vals = get_curve_from_models(norm, only_new)
        plt.errorbar(x, medians, yerr=std_vals, label=norm, capsize=2)

    plt.xlabel("Batch size")
    plt.ylabel("classification error (%)")
    plt.gca().yaxis.grid(True)
    plt.xticks(x, ["32", "16", "8", "4", "2"])
    plt.legend()
    plt.savefig("Images/GN_vs_BN_best_epoch.png", bbox_inches='tight')
