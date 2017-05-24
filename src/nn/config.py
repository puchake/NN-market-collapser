import os

import numpy as np
import tensorflow as tf


class NnConst(object):
    # Global index of performed network run.
    RUN_INDEX = 33

    # Index of input data set, which is used for network
    # (basic_nn_set_0, 1, 2 or 3).
    SET_INDEX = 0

    # Available modes of the network. If it is fresh run (start), continuation
    # of previously started training from the last checkpoint (continue) or
    # usage run (use).
    START_MODE = "start"
    CONTINUE_MODE = "continue"
    USE_MODE = "use"

    # Mode of currently performed run.
    MODE = START_MODE

    # Network's training paths patterns.
    TRAINING_DATA_SET_DIR_PATTERN = "../../data/set/basic_nn_set_{}/"
    TRAIN_LOGS_DIR_PATTERN = "../../data/logs/basic_nn/run_{}/train"
    VALIDATION_LOGS_DIR_PATTERN = "../../data/logs/basic_nn/run_{}/validation"
    CHECKPOINTS_DIR_PATTERN = "../../data/logs/basic_nn/run_{}/checkpoints"
    BEST_MODEL_DIR_PATTERN = "../../data/logs/basic_nn/run_{}/best_model"

    # Paths used in usage runs.
    USAGE_DATA_SET_PATH = "../../data/set/usage/data.npy"
    USAGE_OUT_PATH = "../../data/out/out"
    USAGE_MODEL_PATH_PATTERN = "../../data/logs/basic_nn/run_{}/best_model/" \
                               "best_model"

    # Error message for logs override attempt.
    LOGS_OVERRIDE_ERROR = "Attempted to override existing logs for run {}."

    # Maximum checkpoints to keep. This value will probably be never exceeded.
    MAX_CHECKPOINTS = 1000


class LstmConst(object):
    # Global index of performed network run.
    RUN_INDEX = 33

    # Index of input data set, which is used for network
    # (basic_nn_set_0, 1, 2 or 3).
    SET_INDEX = 0

    # Available modes of the network. If it is fresh run (start), continuation of
    # previously started training from the last checkpoint (continue) or usage run
    # (use).
    START_MODE = "start"
    CONTINUE_MODE = "continue"
    USE_MODE = "use"

    # Mode of currently performed run.
    MODE = START_MODE

    # Network's training paths patterns.
    TRAINING_DATA_SET_DIR_PATTERN = "../../data/set/basic_nn_set_{}/"
    TRAIN_LOGS_DIR_PATTERN = "../../data/logs/basic_nn/run_{}/train"
    VALIDATION_LOGS_DIR_PATTERN = "../../data/logs/basic_nn/run_{}/validation"
    CHECKPOINTS_DIR_PATTERN = "../../data/logs/basic_nn/run_{}/checkpoints"
    BEST_MODEL_DIR_PATTERN = "../../data/logs/basic_nn/run_{}/best_model"

    # Paths used in usage runs.
    USAGE_DATA_SET_PATH = "../../data/set/usage/data.npy"
    USAGE_OUT_PATH = "../../data/out/out"
    USAGE_MODEL_PATH_PATTERN = "../../data/logs/basic_nn/run_{}/best_model/" \
                               "best_model"

    # Error message for logs override attempt.
    LOGS_OVERRIDE_ERROR = "Attempted to override existing logs for run {}."

    # Maximum checkpoints to keep. This value will probably be never exceeded.
    MAX_CHECKPOINTS = 1000


class NnConfig(object):
    TRAIN_BATCH_SIZE = 50000
    NO_DROPOUT = 0.0
    IN_SIZE = 75
    NUM_OF_LABELS = 2
    LAYER_SIZES = [1024, 512, 256, 128, 128, 2]
    DROPOUT = 0.25
    MAX_ITERATIONS = 1000
    VALIDATION_INTERVAL = 10
    LEARNING_RATE = 0.001
    LEARNING_RATE_DECAY = 0.8
    DECAY_INTERVAL = 100
    CHECKPOINT_INTERVAL = 10


class LstmConfig(object):
    UNFOLD_BATCH_SIZE = 50
    TRAIN_BATCH_SIZE = 50000
    NO_DROPOUT = 0.0
    IN_SIZE = 75
    NUM_OF_LABELS = 2
    HIDDEN_SIZE = 512
    FORGET_BIAS = 0.0
    ACTIVATION = tf.tanh
    DROPOUT = 0.25
    MAX_ITERATIONS = 100
    VALIDATION_INTERVAL = 10
    LEARNING_RATE = 0.001
    LEARNING_RATE_DECAY = 0.8
    DECAY_INTERVAL = 100
    CHECKPOINT_INTERVAL = 10


def setup_training_environment(const, run_index, set_index, mode):
    """
    Creates directories necessary for neural network's run. If directories
    already exist, it raises an ValueError to stop user from overriding
    existing logs and models.

    :param run_index: index of the neural network's run
    :param set_index: index of the used data set
    :param mode: current run's mode
    :return: set of paths used by basic nn: data set, train logs,
             validation logs, checkpoints and best model directories
    """

    data_set_dir = const.TRAINING_DATA_SET_DIR_PATTERN.format(set_index)
    train_logs_dir = const.TRAIN_LOGS_DIR_PATTERN.format(run_index)
    validation_logs_dir = const.VALIDATION_LOGS_DIR_PATTERN.format(run_index)
    checkpoints_dir = const.CHECKPOINTS_DIR_PATTERN.format(run_index)
    best_model_dir = const.BEST_MODEL_DIR_PATTERN.format(run_index)

    dirs_to_check = [
        train_logs_dir, validation_logs_dir, checkpoints_dir, best_model_dir
    ]

    for dir_path in dirs_to_check:

        # If one of the log directories already exists raise an error.
        # Else create that directory.
        if os.path.exists(dir_path):
            if mode == const.START_MODE:
                raise ValueError(const.LOGS_OVERRIDE_ERROR.format(run_index))
        else:
            os.makedirs(dir_path)

    return data_set_dir, train_logs_dir, validation_logs_dir, \
           checkpoints_dir, best_model_dir


def load_training_data_set(data_set_dir):
    """
    Load data, which will be used by nn in training, from a specified folder.

    :param data_set_dir: path to folder which contains training data
    :return: train, validation and test data and labels
    """

    train_data = np.load(data_set_dir + "train_data.npy")
    train_labels = np.load(data_set_dir + "train_labels.npy")
    validation_data = np.load(data_set_dir + "validation_data.npy")
    validation_labels = np.load(data_set_dir + "validation_labels.npy")
    test_data = np.load(data_set_dir + "test_data.npy")
    test_labels = np.load(data_set_dir + "test_labels.npy")

    return train_data, train_labels, \
           validation_data, validation_labels, \
           test_data, test_labels
