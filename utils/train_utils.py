import argparse
import os
import shutil
import sys
import time
from tensorflow.keras import datasets
import tensorflow as tf


def get_parsed_in():
    parser = argparse.ArgumentParser(description="Configurations to run normalization experiments with ResNet")
    parser.add_argument('--replace', action='store_true', help='overwrite previous run if it exists')
    parser.add_argument('--cpu', action='store_true', help='train on inly cpu')
    parser.add_argument('--ResNet', type=int, default=20, choices=[20, 32], help='Defines what Resnet model to use')
    parser.add_argument('--batch_size', type=int, default=32, choices=[32, 16, 8, 4, 2], help='batch size per worker')
    parser.add_argument('--epochs', type=int, default=50, choices=[50, 100], help='training epochs')
    parser.add_argument('--norm', type=str, default='GN', choices=['BN', 'GN'], help='decide if BN (batch normalization) or GN (group normalization is applied)')
    parser.add_argument('--run', type=int, default=1, help='Differentiate multiple runs for statistical comparison')


    arguments = vars(parser.parse_args())

    return arguments


def create_folders(arguments):
    path = os.path.join(arguments['norm'], "ResNet" + str(arguments['ResNet']), "batch_" + str(arguments['batch_size']), "run_" + str(arguments['run']))
    log_path = os.path.join("train_outputs/logs", path)
    models_path = os.path.join("train_outputs/checkpoints", path)
    # If paths exist and replacement is desired delete old files
    for i in range(3):
        if os.path.exists(models_path) and os.path.exists(log_path) and arguments['replace']:
            print("Delete old models")
            try:
                shutil.rmtree(models_path)
            except OSError as e:
                print("Error: %s : %s" % (models_path, e.strerror))
                sys.exit(1)

            try:
                shutil.rmtree(log_path)
            except OSError as e:
                print("Error: %s : %s" % (log_path, e.strerror))
                sys.exit(1)

        time.sleep(1)

    # Create file structures
    try:
        os.makedirs(models_path)
    except OSError:
        print("Creation of the directory %s failed, if replacement is desired start script with argument --replace " % models_path)
        sys.exit(1)
    else:
        print("Successfully created the directory %s" % models_path)

    try:
        os.makedirs(log_path)
    except OSError:
        print("Creation of the directory %s failed, if replacement is desired start script with argument --replace " % log_path)
        sys.exit(1)
    else:
        print("Successfully created the directory %s" % log_path)

    return log_path, models_path


class Logger(object):
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(f'{path}/output_log.log', "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def get_data(arguments):
    (train_imgs, train_lbls), (_, _) = datasets.cifar10.load_data()
    train_imgs = train_imgs / 255.0

    split = int(train_imgs.shape[0] * .2)
    x_val = train_imgs[-split:]
    y_val = train_lbls[-split:]
    x_train = train_imgs[:-split]
    y_train = train_lbls[:-split]
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    valid_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    #train_ds = train_ds.map(lambda image, label: (tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)), label)).shuffle(buffer_size=x_train.shape[0])
    train_ds = train_ds.batch(arguments['batch_size']).prefetch(2)
    valid_ds = valid_ds.batch(arguments['batch_size']).prefetch(2)
    return  train_ds, valid_ds


