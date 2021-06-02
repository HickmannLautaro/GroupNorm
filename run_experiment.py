import os
import shutil
from datetime import datetime

import numpy as np
import tensorflow as tf

from utils.ResNet_network import get_resnet_n, load_model_with_optimizer, save_model
from utils.train_utils import get_parsed_in, create_folders, get_data, select_device


class CustomSaveModel(tf.keras.callbacks.Callback):
    """
    Custom callback to save the model weights and optimizer when the validation accuracy improves
    """

    def __init__(self, path, accu):
        """
        Initialization method
        @param path: where to save the weights and optimizer
        @param accu: when continuing the training from a checkpoint set the initial accuracy to the one achieved on it.
        """
        super(CustomSaveModel, self).__init__()
        self.path = path
        self.best_accuracy = accu

    def on_train_begin(self, logs=None):
        print(f"Starting accuracy: {np.around(self.best_accuracy, 6)}")

    def on_epoch_end(self, epoch, logs=None):
        """
        If the val accuracy improved save the weights and optimizer to the corresponding folder
        @param epoch: current epoch
        @param logs: current metrics
        """
        current = logs.get("val_sparse_categorical_accuracy")
        if np.greater(current, self.best_accuracy):  # Check for improvements
            save_model(self.path, self.model, epoch)  # Call the custom save function
            print(f"Saving model accuracy improved from {self.best_accuracy} to {current}")
            self.best_accuracy = current
        else:
            print(f"model accuracy did not improved from {self.best_accuracy}")


def get_scheduler(arguments):
    """
    Returns a learning rate scheduler depending on the training epochs.
    Since in the data Augmentation the repeat 4 makes the data 4 times bigger, the steps of 100 epochs would be done in 25
    @param arguments: training arguments for this function for how many epochs will be trained is needed.
    @return: a learning rate scheduler that divides the current lr by 10 at three points.
    """

    if arguments['epochs'] == 30:
        def scheduler_30(epoch, lr):
            if epoch == 8:  # 30/4 ~ 8
                lr = lr / 10.0
            elif epoch == 15:  # 60/4 ~ 15
                lr = lr / 10.0
            elif epoch == 22:  # 90/4 ~ 22
                lr = lr / 10.0

            print("epoch :", epoch + 1, " lr :", np.around(lr, 6))
            return lr

        return scheduler_30

    else:
        def scheduler_100(epoch, lr):

            if epoch == 30:  # 30/4 ~ 8
                lr = lr / 10.0
            elif epoch == 60:  # 60/4 ~ 15
                lr = lr / 10.0
            elif epoch == 90:  # 90/4 ~ 22
                lr = lr / 10.0

            print("epoch :", epoch + 1, " lr :", np.around(lr, 6))
            return lr

        return scheduler_100


def main():
    # Get command line arguments
    arguments = get_parsed_in()

    # Create needed folder structure
    log_path, models_path = create_folders(arguments)

    # Select witch device to train on CPU or GPU
    device = select_device(arguments)

    print("Tensorflow version: ", tf.__version__)

    # Get training and validation data
    train_data, validation_data = get_data(arguments)

    # Continue training from checkpoint or start new training
    if arguments['continue']:
        latest = os.listdir(models_path)  # Search for checkpoints
        if not latest:  # If no checkpoint is found create new model
            mod = "Tried to load but none found, creating new model"
            model, init_ep = get_resnet_n(arguments)
            best_accu = 0
        else:  # From the checkpoint list get the latest (best) and load the weights
            latest.sort()  # Latter checkpoints are better
            latest = latest[-1]
            init_ep = int(latest[-4:])  # Get staring epoch from checkpoint name
            if arguments['cont_epoch'] != -1:  # Overwrite starting epoch for continued training with the one provided in the configurations
                init_ep = arguments['cont_epoch']
            mod = f"Loading model from {latest}, starting epoch {init_ep}"
            model = load_model_with_optimizer(arguments, models_path + '/' + latest)  # Custom function to load model weights and optimizer
            best_accu = model.evaluate(validation_data)[1]  # Get the best accuracy for the save model callback
    else:  # Start new training
        mod = "New model"
        model, init_ep = get_resnet_n(arguments)  # Get new model
        best_accu = 0

    # Callbacks
    scheduler_callback = tf.keras.callbacks.LearningRateScheduler(get_scheduler(arguments))
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(log_path, 'training.log'))
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=10, write_graph=False, profile_batch=0)
    custom_save = CustomSaveModel(models_path, best_accu)

    # Print training information
    print(f"Experiment ResNet{arguments['ResNet'] * 6 + 2} on CIFAR10, config:"
          f"\n-----------------------------------------------------------------"
          f"\nRunning on: {device}"
          f"\nmodel: {mod}"
          f"\nbatch_size: {arguments['batch_size']}"
          f"\nepochs: {arguments['epochs']}"
          f"\nnormalization: {arguments['norm']}"
          f"\nUse weight decay: {arguments['weight_decay']}"
          f"\nTraining log in: {os.path.join(log_path, 'training.log')}"
          f"\nTraining checkpoints in: {models_path}"
          f"\nTraining started at: {str(datetime.now())}"
          f"\n-----------------------------------------------------------------")

    # Validation on the test set as proposed in the ResNet paper Deep Residual Learning for Image Recognition (https://arxiv.org/pdf/1512.03385.pdf)
    model.fit(train_data, epochs=arguments['epochs'], validation_data=validation_data, initial_epoch=init_ep, callbacks=[scheduler_callback, csv_logger, tensorboard, custom_save])  # Fromm 100 to 30 (25*4 + 5 extra) epochs

    # delete unused checkpoints (to save space)
    chkpts = os.listdir(models_path)  # Get all generated checkpoints.
    chkpts.sort()
    latest = chkpts[-1]  # The last checkpoint is the best one.
    chkpts = [os.path.join(models_path, x) for x in chkpts]  # Get path from root to checkpoint for all found checkpoints
    to_delete = [x for x in chkpts if latest not in x]  # Filter the latest checkpoint from the to delete list
    for file in to_delete:
        print("Removed : ", file)
        shutil.rmtree(file)


if __name__ == "__main__":
    main()
