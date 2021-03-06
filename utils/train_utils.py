import argparse
import os
import shutil
import sys
import time
import tensorflow as tf
import tensorflow_datasets as tfds


def get_parsed_in():
    """
    Parse command line inputs
    @return: arguments dictionary containing parsed command line inputs
    """
    parser = argparse.ArgumentParser(description="Configurations to run normalization experiments with ResNet")
    parser.add_argument('--replace', action='store_true', help='overwrite previous run if it exists')
    parser.add_argument('--continue', action='store_true', help='continue training: load the latest checkpoint and continue training')
    parser.add_argument('--cont_epoch', type=int, default=-1, help='Used together with continue, overwrites the saved epoch of the checkpoint and sets the initial epoch for continue training')
    parser.add_argument('--experimental_data_aug', action='store_true', help='Use experimental data augmentation inside the network instead of the one before the network (Only used for epochs = 100)')
    parser.add_argument('--cpu', action='store_true', help='train on only cpu')
    parser.add_argument('--ResNet', type=int, default=3, choices=[3, 5], help='Defines what Resnet model to use, 3-> ResNet 20, 5-> ResNet 32')
    parser.add_argument('--batch_size', type=int, default=32, choices=[32, 16, 8, 4, 2], help='batch size per worker')
    parser.add_argument('--epochs', type=int, default=30, choices=[30, 100], help='training epochs')
    parser.add_argument('--norm', type=str, default='GN', choices=['BN', 'GN'], help='decide if BN (batch normalization) or GN (group normalization is applied)')
    parser.add_argument('--run', type=int, default=1, help='Differentiate multiple runs for statistical comparison')
    parser.add_argument('--weight_decay', action='store_true', help='Set to use SGDW (stochastic gradient with weight decay) as optimizer and use weight decay (unstable) otherwise adam is used.')

    arguments = vars(parser.parse_args())

    if arguments['continue'] and arguments['replace']:
        print("Incompatible options to continue training and remove it to replace choose one or the other")
        sys.exit()
    return arguments


def create_folders(arguments):
    """
    Creates the folder structure needed to save logs and checkpoints depending on the
    @param arguments: run configuration
    @return: log_path where to save logs and models_path where to save the checkpoints.
    """
    if arguments['weight_decay']:
        path = os.path.join(arguments['norm'], "ResNet" + str(arguments['ResNet'] * 6 + 2), "weight_decay", "epochs_" + str(arguments['epochs']), "batch_" + str(arguments['batch_size']), "run_" + str(arguments['run']))
    else:
        path = os.path.join(arguments['norm'], "ResNet" + str(arguments['ResNet'] * 6 + 2), "no_weight_decay", "epochs_" + str(arguments['epochs']), "batch_" + str(arguments['batch_size']), "run_" + str(arguments['run']))

    log_path = os.path.join("train_outputs/logs", path)
    models_path = os.path.join("train_outputs/checkpoints", path)

    # If paths exist and replacement is desired delete old files. To solve interference with tensorboard try to delete multiple times. Probably better solutions exist.
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
    """
    Data augmentation function, random zoom a part of an image and then resize to 32,32,3
    @param image: input image
    @return: zoomed image.
    """
    sz = tf.random.uniform((1,), 24, 32, dtype=tf.int32)  # Get random state
    zoom = tf.image.random_crop(image, size=(sz[0], sz[0], 3))  # Crop image
    zoomed = tf.image.resize(zoom, (32, 32), method=tf.image.ResizeMethod.BICUBIC)  # resize image to correct dimensions
    return zoomed


@tf.function
def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.image.convert_image_dtype(image, tf.float32), label  # tf.cast(image, tf.float32) / 255.0, label  # tf.image.convert_image_dtype(image, tf.float32), label


def prepare_dataset(image, label):
    """
    Apply data augmentation to
    @param image: input image
    @param label: corresponding label
    @return: augmented image and original label
    """
    image = tf.image.random_flip_left_right(image)
    image = random_zoom(image)
    return image, label


def get_data(arguments):
    """
    Load training and validation data and apply normalization and optionally data augmentation
    @param arguments: batch size
    @return: training and validation sets
    """
    batch_size = arguments['batch_size']

    (train, test), ds_info = tfds.load('cifar10', split=['train', 'test'], as_supervised=True, with_info=True) # Load CIFAR-10

    buffer_size = 10000

    if arguments['epochs'] == 30:
        # For better results and since map is not really random (https://github.com/tensorflow/tensorflow/issues/35682#issuecomment-573777790)
        # the data is repeated 4 times and therefore the training epochs are reduced from 100 to 25, 30 was used because I mistyped and realized to late.
        train = train.map(normalize_img).repeat(4).map(prepare_dataset).batch(batch_size).shuffle(buffer_size).cache()
    else:
        train = train.map(normalize_img).batch(batch_size).shuffle(buffer_size).cache()

    # Normalize the validation/test set
    test = test.map(normalize_img).batch(batch_size)

    return train, test


def get_test_data(arguments):
    """
    Load and normalize only the test dataset
    @param arguments: batch size
    @return: test set
    """
    batch_size = arguments['batch_size']

    (train, test), ds_info = tfds.load('cifar10', split=['train', 'test'], as_supervised=True, with_info=True)

    test = test.map(normalize_img).batch(batch_size)

    return test


def select_device(arguments):
    """
    Sets if training should happen on CPU or GPU
    @param arguments: CPU or GPU
    @return: string for information print
    """
    if arguments['cpu']:
        device = 'CPU'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Make GPU not visible for tensorflow
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
