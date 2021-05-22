import tensorflow as tf
import sys
from datetime import datetime
from utils.train_utils import get_parsed_in, create_folders





def main():
    # Get command line arguments
    arguments = get_parsed_in()

    # Create needed folder structure
    log_path, models_path = create_folders(arguments)

    # Log output
    sys.stdout = open(f'{log_path}/output_log.log', 'w')
    sys.stderr = open(f'{log_path}/output_log.log', 'w')
    print("Training started: "+ str(datetime.now()))

    # Turn memory growth an to run multiple scripts in parallel on the same GPU
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    print("Tensorflow version: ", tf.__version__)
    print("Training config: "+ str(arguments))


    # TODO delete unused checkpoints
    sys.stdout.close()


if __name__ == "__main__":
    main()
