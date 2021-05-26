import argparse
import os
import shutil
import sys
import time
from tensorflow.keras import datasets
import tensorflow as tf
import tensorflow_datasets as tfds


def get_parsed_in():
    parser = argparse.ArgumentParser(description="Configurations to run normalization experiments with ResNet")
    parser.add_argument('--replace', action='store_true', help='overwrite previous run if it exists')
    parser.add_argument('--continue', action='store_true', help='continue training: load the latest checkpoint and continue training')
    parser.add_argument('--cont_epoch', type=int, default=-1, help='Used together with continue, overwrites the saved epoch of the checkpoint and sets the initial epoch for continue training')
    parser.add_argument('--cpu', action='store_true', help='train on inly cpu')
    parser.add_argument('--ResNet', type=int, default=3, choices=[3, 5], help='Defines what Resnet model to use, 3-> ResNet 20, 5-> ResNet 32')
    parser.add_argument('--batch_size', type=int, default=32, choices=[32, 16, 8, 4, 2], help='batch size per worker')
    parser.add_argument('--epochs', type=int, default=30, choices=[30, 100], help='training epochs')
    parser.add_argument('--norm', type=str, default='GN', choices=['BN', 'GN'], help='decide if BN (batch normalization) or GN (group normalization is applied)')
    parser.add_argument('--run', type=int, default=1, help='Differentiate multiple runs for statistical comparison')
    parser.add_argument('--weight_decay', action='store_true', help='Set to use AdamW as optimizer and use weight decay (unstable)')

    arguments = vars(parser.parse_args())


    if arguments['continue'] and arguments['replace']:
        print("Incompatible options to continue training and remove it to replace choose one or the other")
        sys.exit()
    return arguments


def create_folders(arguments):
    if arguments['weight_decay']:
        path = os.path.join(arguments['norm'], "ResNet" + str(arguments['ResNet']*6+2),"weight_decay" ,"epochs_"+ str(arguments['epochs']),"batch_" + str(arguments['batch_size']), "run_" + str(arguments['run']))
    else:
        path = os.path.join(arguments['norm'], "ResNet" + str(arguments['ResNet']*6+2),"no_weight_decay","epochs_"+ str(arguments['epochs']),"batch_" + str(arguments['batch_size']), "run_" + str(arguments['run']))

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
        if not arguments['continue']:
            print("Creation of the directory %s failed, if replacement is desired start script with argument --replace " % models_path)
            sys.exit(1)
    else:
        print("Successfully created the directory %s" % models_path)

    try:
        os.makedirs(log_path)
    except OSError:
        print("Creation of the directory %s failed" % log_path)
        if not arguments['continue']:
            print("Creation of the directory %s failed, if replacement is desired start script with argument --replace " % log_path)
            sys.exit(1)
    else:
        print("Successfully created the directory %s" % log_path)

    return log_path, models_path


@tf.function
def random_zoom(image):
    sz = tf.random.uniform((1,), 24, 32, dtype=tf.int32)
    zoom = tf.image.random_crop(image, size=(sz[0], sz[0], 3))
    zoomed = tf.image.resize(zoom, (32, 32), method=tf.image.ResizeMethod.BICUBIC)
    return zoomed


@tf.function
def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.image.convert_image_dtype(image, tf.float32), label  # tf.cast(image, tf.float32) / 255.0, label  # tf.image.convert_image_dtype(image, tf.float32), label


def prepare_dataset(image, label):
    image = tf.image.random_flip_left_right(image)
    image = random_zoom(image)
    return image, label



def get_data(arguments):
    batch_size = arguments['batch_size']

    (ds_train, ds_test), ds_info = tfds.load(
        'cifar10',
        split=['train', 'test'],
        as_supervised=True,
        with_info=True,
    )
    BUFFER_SIZE = 10000
    if arguments['epochs'] == 30:
        ds_train = ds_train.map(normalize_img).repeat(4).map(prepare_dataset).batch(batch_size).shuffle(BUFFER_SIZE).cache()
    else:
        ds_train = ds_train.map(normalize_img).map(prepare_dataset).batch(batch_size).shuffle(BUFFER_SIZE).cache()

    ds_test = ds_test.map(normalize_img).batch(batch_size)
    return ds_train, ds_test


def select_device(arguments):
    if arguments['cpu']:
        device = 'CPU'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        device = 'GPU'
        # Turn memory growth om to run multiple scripts in parallel on the same GPU
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(gpu_devices) > 0:
            tf.config.experimental.set_memory_growth(gpu_devices[0], True)
        else:
            print("GPU selected but no GPUs available")
            sys.exit(1)
    return device


