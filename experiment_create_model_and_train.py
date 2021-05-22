import tensorflow as tf
import sys
from datetime import datetime
from utils.train_utils import get_parsed_in, create_folders


def group_norm(x, gamma, beta, G, eps=0.00001):
    # Watch channel last vs channel first(paper)
    # From NHWC to NCHW
    x = tf.transpose(x, [0, 3, 1, 2])

    # x: input features with shape [N,C,H,W]
    # gamma, beta: scale and offset, with shape [1,C,1,1]
    # G: number of groups for GN
    N, C, H, W = x.shape
    x = tf.reshape(x, [N, G, C // G, H, W])
    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
    x = (x - mean) / tf.sqrt(var + eps)
    x = tf.reshape(x, [N, C, H, W])
    x = x * gamma + beta

    # From NCHW to NHWC
    return tf.transpose(x, [0, 2, 3, 1])


def get_resnet(arguments):
    # TODO create resnet depending on arguments
    # TODO add data augmentation layers
    return 1


def make_or_load_model(arguments, models_path):
    # TODO add method to load from checkpoint
    # if arguments['continue_train']:
    #     checkpoints = [models_path + "/" + name for name in os.listdir(models_path)]
    #     if checkpoints:
    #         latest_checkpoint = max(checkpoints, key=os.path.getctime)
    #         print("Restoring from", latest_checkpoint)
    #         return keras.models.load_model(latest_checkpoint)

    # TODO create model
    print("Creating a new model")
    return get_resnet(arguments)


def main():
    # Get command line arguments
    arguments = get_parsed_in()

    # Create needed folder structure
    log_path, models_path = create_folders(arguments)

    # Log output
    sys.stdout = open(f'{log_path}/output_log.log', 'w')
    sys.stderr = open(f'{log_path}/output_log.log', 'w')
    print("Training started: " + str(datetime.now()))

    # Turn memory growth an to run multiple scripts in parallel on the same GPU
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    print("Tensorflow version: ", tf.__version__)
    print("Training config: " + str(arguments))

    # TODO create or load model
    model = make_or_load_model(arguments, models_path)

    # TODO load dataset

    # TODO train model

    # TODO delete unused checkpoints
    sys.stdout.close()


if __name__ == "__main__":
    main()
