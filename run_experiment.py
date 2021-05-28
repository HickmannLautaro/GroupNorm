import pickle

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import os
from utils.train_utils import get_parsed_in, create_folders, get_data, select_device
from utils.ResNet_network import get_resnet_n, load_model_with_optimizer, save_model
import shutil
from datetime import datetime


class CustomSaveModel(tf.keras.callbacks.Callback):
    def __init__(self, path, accu):
        super(CustomSaveModel, self).__init__()
        self.path = path
        self.best_accuracy = accu

    def on_train_begin(self, logs=None):
        print(f"Strating accuracy: {np.around(self.best_accuracy, 6)}")

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("val_sparse_categorical_accuracy")
        if np.greater(current, self.best_accuracy):
            save_model(self.path, self.model, epoch)
            print(f"Saving model accuracy improved from {self.best_accuracy} to {current}")
            self.best_accuracy = current
        else:
            print(f"model accuracy did not improved from {self.best_accuracy}")


def get_scheduler(arguments):
    # Since in the data Augmentation repeat 4, the steps of 100 epochs would be done in 25
    # After trying multiple stuff this works the best
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

    device = select_device(arguments)

    print("Tensorflow version: ", tf.__version__)

    train, eval = get_data(arguments)

    if arguments['continue']:

        latest = os.listdir(models_path)  # tf.train.latest_checkpoint(models_path)
        if not latest:
            mod = "Tried to load but none found, creating new model"
            model, init_ep = get_resnet_n(arguments)
            best_accu = 0
        else:
            latest.sort()
            latest = latest[-1]
            init_ep = int(latest[-4:])
            if arguments['cont_epoch'] != -1:
                init_ep = arguments['cont_epoch']
            mod = f"Loading model from {latest}, starting epoch {init_ep}"
            model = load_model_with_optimizer(arguments, models_path + '/' + latest)
            best_accu = model.evaluate(eval)[1]
    else:
        mod = "New model"
        model, init_ep = get_resnet_n(arguments)
        best_accu = 0

    scheduler_callback = tf.keras.callbacks.LearningRateScheduler(get_scheduler(arguments))
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(log_path, 'training.log'))
    csv_logger_evaluation = tf.keras.callbacks.CSVLogger(os.path.join(log_path, 'evaluate.log'))
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=10, write_graph=False, profile_batch=0)

    custom_save = CustomSaveModel(models_path, best_accu)

    print(f"Experiment ResNet{arguments['ResNet'] * 6 + 2} on CIFAR10, config:"
          f"\n-----------------------------------------------------------------"
          f"\nRunning on: {device}"
          f"\nmodel: {mod}"
          f"\nbatch_size: {arguments['batch_size']}"
          f"\nepochs: {arguments['epochs']}"
          f"\nnormalization: {arguments['norm']}"
          f"\nUse weight decay: {arguments['weight_decay']}"
          f"\nTrainning log in: {os.path.join(log_path, 'training.log')}"
          f"\nTrainning checkpoints in: {models_path}"
          f"\nTraining started at: {str(datetime.now())}"
          f"\n-----------------------------------------------------------------")

    # Validation on the test set as proposed in the ResNet paper Deep Residual Learning for Image Recognition (https://arxiv.org/pdf/1512.03385.pdf)
    model.fit(train, epochs=arguments['epochs'], validation_data=eval, initial_epoch=init_ep, callbacks=[scheduler_callback, csv_logger, tensorboard, custom_save])  # Fromm 100 to 30 (25*4 + 5 extra )epochs

    # delete unused checkpoints
    chkpts = os.listdir(models_path)
    chkpts.sort()
    latest = chkpts[-1]
    chkpts = [os.path.join(models_path, x) for x in chkpts]
    to_delete = [x for x in chkpts if not latest in x]
    for file in to_delete:
        print("Removed : ", file)
        shutil.rmtree(file)


if __name__ == "__main__":
    main()
