import argparse
import os
import shutil
import sys


def get_parsed_in():
    parser = argparse.ArgumentParser(description="Configurations to run normalization experiments with ResNet")
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest checkpoint and continue training')
    parser.add_argument('--replace', action='store_true', help='overwrite previous run if it exists')
    parser.add_argument('--start_epoch', type=int, default=1, help='Differentiate multiple runs for statistical comparison')
    parser.add_argument('--ResNet', type=int, default=18, choices=[18, 34], help='Defines what Resnet model to use')
    parser.add_argument('--batch_size', type=int, default=32, choices=[32, 16, 8, 4, 2], help='batch size per worker')
    parser.add_argument('--norm', type=str, default='GN', choices=['BN', 'GN'], help='decide if BN (batch normalization) or GN (group normalization is applied)')
    parser.add_argument('--run', type=int, default=1, help='Differentiate multiple runs for statistical comparison')

    arguments = vars(parser.parse_args())
    if arguments['continue_train'] and arguments['replace']:
        print("Incompatible options to continue training and remove it to replace choose one or the other")
        sys.exit()
    return arguments


def create_folders(arguments):
    path = os.path.join(arguments['norm'], "ResNet" + str(arguments['ResNet']), "batch_" + str(arguments['batch_size']), "run_" + str(arguments['run']))
    log_path = os.path.join("train_outputs/logs", path)
    models_path = os.path.join("train_outputs/checkpoints", path)
    # If paths exist and replacement is desired delete old files
    if os.path.exists(models_path) and os.path.exists(log_path) and arguments['replace']:
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
