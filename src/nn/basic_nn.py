import numpy as np
import tensorflow as tf

from src.nn.config import NnConfig as config
from src.nn.config import NnConst as const
from src.nn.config import setup_training_environment, load_training_data_set


def create_fully_connected_layer(
        name, in_node, in_size, num_of_neurons,
        is_output_layer, dropout
):
    """
    Create fully connected layer's variables and use them to transform in-node
    into out-node with optional activation and specified dropout. There is no
    need to add activation rescaling for no-dropout case, because tensorflow
    rescales outputs with dropout automatically (scales up by 1 / keep_prob).

    :param name: name which will be given to the layer
    :param in_node: input tensorflow graph node (vector)
    :param in_size: size of the input vector
    :param num_of_neurons: number of neurons for created layer
    :param is_output_layer: states whether created layer is network's last
                            layer. In this case no activation function is
                            used
    :param dropout: probability of dropout for this layer (used in training)
    :return:
    """

    weights = tf.get_variable(
        name + "/weights",
        shape=[in_size, num_of_neurons],
        initializer=tf.contrib.layers.xavier_initializer()
    )
    biases = tf.get_variable(
        name + "/biases",
        shape=[1, num_of_neurons],
        initializer=tf.constant_initializer()
    )

    out = tf.matmul(in_node, weights) + biases

    # Add histograms.
    tf.summary.histogram(weights.op.name, weights)
    tf.summary.histogram(biases.op.name, biases)
    tf.summary.histogram(out.op.name, out)

    if is_output_layer:
        return tf.nn.dropout(out, 1 - dropout)
    else:
        return tf.nn.dropout(tf.maximum(out, out * 0.1), 1 - dropout)


def create_graph(
        in_placeholder, in_size, layer_sizes,
        labels, learning_rate_placeholder, dropout_placeholder
):
    """
    Creates graph for basic neural network both for training and teest case.

    :param in_placeholder: placeholder for input data batch
    :param in_size: size of input vector
    :param layer_sizes: list of neurons count for each layer
    :param labels: set of labels used to evaluate network
    :param learning_rate_placeholder: placeholder for network run's learning
                                      rate
    :param dropout_placeholder: placeholder for network's dropout probability
    :return: created out, loss, train and accuracy nodes
    """

    # Create network graph.
    previous_in = in_placeholder
    previous_in_size = in_size
    for layer_size, i in zip(layer_sizes, range(len(layer_sizes))):
        previous_in = create_fully_connected_layer(
            "layer_{}".format(i),
            previous_in, previous_in_size,
            layer_size,
            i == len(layer_sizes) - 1,
            dropout_placeholder
        )
        previous_in_size = layer_size

    # Create out, loss, train and accuracy nodes.
    out = tf.nn.softmax(previous_in)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=previous_in
        )
    )
    train = tf.train.AdamOptimizer(
        learning_rate_placeholder
    ).minimize(loss)
    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.argmax(previous_in, axis=1), tf.argmax(labels, axis=1)
            ),
            tf.float32
        )
    )

    return out, loss, train, accuracy


def perform_usage_run(run_index, data_set_path, out_path):
    """
    Perform single network's forward pass by using best model saved for given
    run index. Save results to output file on given path.

    :param run_index: index of run which will be source for used model
    :param data_set_path: path to data set file which will be processed
    :param out_path: path to file which will contain forward pass results
    :return: -
    """

    # Fill in usage model path pattern.
    usage_model_path = const.USAGE_MODEL_PATH_PATTERN.format(run_index)

    # Load data.
    data = np.load(data_set_path)

    # Create network's placeholders.
    in_placeholder = tf.placeholder(tf.float32, shape=[None, config.IN_SIZE])
    labels_placeholder = tf.placeholder(
        tf.float32, shape=[None, config.NUM_OF_LABELS]
    )
    learning_rate_placeholder = tf.placeholder(tf.float32, shape=[])
    dropout_placeholder = tf.placeholder(tf.float32, shape=[])

    # Create network's graph.
    out, _, _, _ = create_graph(
        in_placeholder, config.IN_SIZE, config.LAYER_SIZES, labels_placeholder,
        learning_rate_placeholder, dropout_placeholder
    )

    # Create session and initialize all variables.
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # Create model saver and restore best model.
    saver = tf.train.Saver()
    saver.restore(session, usage_model_path)

    # Perform one forward pass.
    calculated_out = session.run(
        [out],
        feed_dict={
            in_placeholder: data, dropout_placeholder: config.NO_DROPOUT
        }
    )

    # Save results to file and exit.
    np.save(out_path, calculated_out)


def train_network(run_index, set_index, mode, learning_rate):
    """
    Start/continue basic nn training for given run index and with data set
    described by set index.

    :param run_index:
    :param set_index:
    :param mode:
    :param learning_rate:
    :return:
    """

    # Setup training environment.
    data_set_dir, train_logs_dir, validation_logs_dir, \
    checkpoints_dir, best_model_dir = setup_training_environment(
        const, run_index, set_index, mode
    )

    train_data, train_labels, \
    validation_data, validation_labels, \
    test_data, test_labels = load_training_data_set(data_set_dir)

    # Create network's placeholders.
    in_placeholder = tf.placeholder(tf.float32, shape=[None, config.IN_SIZE])
    labels_placeholder = tf.placeholder(
        tf.float32, shape=[None, config.NUM_OF_LABELS]
    )
    learning_rate_placeholder = tf.placeholder(tf.float32, shape=[])
    dropout_placeholder = tf.placeholder(tf.float32, shape=[])

    # Create network's graph.
    out, loss, train, accuracy = create_graph(
        in_placeholder, config.IN_SIZE, config.LAYER_SIZES, labels_placeholder,
        learning_rate_placeholder, dropout_placeholder
    )

    # Create rest of summary and train, and validation logs writers.
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary = tf.summary.merge_all()
    train_logger = tf.summary.FileWriter(train_logs_dir)
    validation_logger = tf.summary.FileWriter(validation_logs_dir)

    # Create session and initialize all variables.
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    lowest_loss = None

    if config.MODE == config.CONTINUE_MODE:
        # Might be not needed.
        pass

    # Create model saver which will keep all checkpoints.
    saver = tf.train.Saver(max_to_keep=config.MAX_CHECKPOINTS)

    for i in range(config.MAX_ITERATIONS):

        # Sample train data.
        batch_indices = np.random.permutation(train_data.shape[0])
        batch_indices = batch_indices[:config.TRAIN_BATCH_SIZE]
        batch_data = train_data[batch_indices]
        batch_labels = train_labels[batch_indices]

        # Perform normal train run.
        summary, _ = session.run(
            [merged_summary, train],
            feed_dict={
                in_placeholder: batch_data,
                labels_placeholder: batch_labels,
                learning_rate_placeholder: learning_rate,
                dropout_placeholder: config.DROPOUT
            }
        )

        train_logger.add_summary(summary, i)

        # Save checkpoint for model.
        if i % config.CHECKPOINT_INTERVAL == 0:
            saver.save(session, checkpoints_dir + "/model_checkpoint", i)

        # Validation run.
        if i % config.VALIDATION_INTERVAL == 0:
            summary, loss_value, accuracy_value = session.run(
                [merged_summary, loss, accuracy],
                feed_dict={
                    in_placeholder: validation_data,
                    labels_placeholder: validation_labels,
                    dropout_placeholder: config.NO_DROPOUT
                }
            )

            # If we obtained better validation loss, save the model.
            if lowest_loss is None or loss_value < lowest_loss:
                lowest_loss = loss_value
                saver.save(session, best_model_dir + "/best_model")

            validation_logger.add_summary(summary, i)

            # Also output some data to console.
            print(
                "Iteration {}:\n\tLoss: {}\n\tAccuracy: {}\n".format(
                    i, loss_value, accuracy_value
                )
            )

        # Learning rate decay.
        if i % config.DECAY_INTERVAL == 0 and i != 0:
            learning_rate *= config.LEARNING_RATE_DECAY

    # Test run.
    saver.save(session, checkpoints_dir + "/model_checkpoint_last")
    saver.restore(session, best_model_dir + "/best_model")
    loss_value, accuracy_value = session.run(
        [loss, accuracy],
        feed_dict={
            in_placeholder: test_data,
            labels_placeholder: test_labels,
            dropout_placeholder: config.NO_DROPOUT
        }
    )
    print(
        "Final Loss: {}\n\tFinal Accuracy: {}\n".format(
            loss_value, accuracy_value
        )
    )


def main(MODE):
    """
    Main function of this script.
    :param MODE: run mode
    :return: -
    """
    if MODE == const.USE_MODE:
        perform_usage_run(
            const.RUN_INDEX, const.USAGE_DATA_SET_PATH, const.USAGE_OUT_PATH
        )
    else:
        train_network(
            const.RUN_INDEX, const.SET_INDEX, const.MODE, config.LEARNING_RATE
        )


if __name__ == "__main__":
    main(const.USE_MODE)
