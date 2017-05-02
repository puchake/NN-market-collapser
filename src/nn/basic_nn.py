import tensorflow as tf
import numpy as np


DATA_SET_FOLDER = "../../data/set/first_nn/"
USAGE_DROPOUT = 0.0


def load_data_set(data_set_folder):
    """
    Load data, which will be used by nn in training, from a specified folder.

    :param data_set_folder: path to folder which contains training data
    :return: train, validation and test data and labels
    """

    train_data = np.load(data_set_folder + "train_data.npy")
    train_labels = np.load(data_set_folder + "train_labels.npy")
    validation_data = np.load(data_set_folder + "validation_data.npy")
    validation_labels = np.load(data_set_folder + "validation_labels.npy")
    test_data = np.load(data_set_folder + "test_data.npy")
    test_labels = np.load(data_set_folder + "test_labels.npy")

    return train_data, train_labels, \
           validation_data, validation_labels, \
           test_data, test_labels


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
    bias = tf.get_variable(
        name + "/bias",
        shape=[1, num_of_neurons],
        initializer=tf.constant_initializer()
    )

    out = tf.matmul(in_node, weights) + bias

    if is_output_layer:
        return tf.nn.dropout(out, 1 - dropout)
    else:
        return tf.nn.dropout(tf.nn.relu(out), 1 - dropout)


def create_graph(
        in_placeholder, in_size, layer_sizes,
        for_training, labels=None,
        learning_rate_placeholder=None, dropout_placeholder=None
):
    """
    Creates graph for basic neural network both for training and use case.

    :param in_placeholder: placeholder for input data batch
    :param in_size: size of input vector
    :param layer_sizes: list of neurons count for each layer
    :param for_training: states if the graph will be used in model's training
    :param labels: set of labels used to evaluate network
    :param learning_rate_placeholder: placeholder for network run's learning
                                      rate
    :param dropout_placeholder: placeholder for network's dropout probability
    :return: created out or loss, train and accuracy nodes
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
            dropout_placeholder if for_training else USAGE_DROPOUT
        )
        previous_in_size = layer_size

    if not for_training:

        # Return last output with softmax on top of it.
        return tf.nn.softmax(previous_in_size)

    else:

        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=previous_in
        )
        train = tf.train.AdamOptimizer(
            learning_rate_placeholder
        ).minimize(loss)
        accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(previous_in, axis=1), labels), tf.float32
            )
        )

        return loss, train, accuracy

if __name__ == "__main__":
    pass