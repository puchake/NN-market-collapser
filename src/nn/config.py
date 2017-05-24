class Const(object):
    # Global index of performed network run.
    RUN_INDEX = 32

    # Index of input data set, which is used for network
    # (basic_nn_set_0, 1 or 2).
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
    danę=10

class LstmConfig(object):
    UNFOLD_BATCH_SIZE = 50
    TRAIN_BATCH_SIZE = 50000
    NO_DROPOUT = 0.0
    IN_SIZE = 75
    NUM_OF_LABELS = 2
    LAYER_SIZES = [512, 256, 2]
    DROPOUT = 0.25
    MAX_ITERATIONS = 100
    VALIDATION_INTERVAL = 10
    LEARNING_RATE = 0.001
    LEARNING_RATE_DECAY = 0.8
    DECAY_INTERVAL = 100
    CHECKPOINT_INTERVAL = 10
    danę = 10