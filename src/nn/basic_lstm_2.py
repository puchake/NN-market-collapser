import tensorflow as tf
import numpy as np
from os import listdir


# Graph parameters.
HIDDEN_SIZE = 512
ROLL_OUT = 50
NUM_LAYERS = 4
NUM_CLASSES = 2
# In size with 6 values values day.
IN_SIZE = 6
# Constant bounds for the gradient in backward pass.
GRAD_MIN = -5.0
GRAD_MAX = 5.0
# Learning parameters.
BATCH_SIZE = 200
MAX_ITERATIONS = 1000000
SAMPLE_INTERVAL = 100
START_LEARNING_RATE = 0.1


def create_cell(hidden_size, num_classes, num_layers):
    """
    Create whole lstm rnn cell with output layer on top of it and return
    created variables.
    :param hidden_size: size of the hidden layers
    :param num_classes: desired size of the output
    :param num_layers: number of used layers
    :return: created rnn cell and output transformation
    """
    # Create multi layer rnn cell.
    cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
    multi_cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
    # Create output transformation.
    out_weights = tf.get_variable(
        "out_weights",
        shape=[hidden_size, num_classes],
        initializer=tf.contrib.layers.xavier_initializer()
    )
    out_biases = tf.get_variable(
        "out_biases",
        shape=[num_classes, ],
        initializer=tf.constant_initializer()
    )
    return multi_cell, out_weights, out_biases


def create_train_graph(
    cell, out_weights, out_biases, num_layers, num_classes, roll_out,
    in_data, prev_state, labels, learning_rate
):
    """
    Map out the training graph with previously created cell and output
    transformation.
    :param cell: previously created tensorflow cell
    :param out_weights: output weights matrix
    :param out_biases: output biases vector
    :param num_layers: number of used lstm layers
    :param roll_out: length of the training graph
    :param in_data: input data placeholder of size
                    roll_out x batch_size x in_size
    :param prev_state: previous state placeholder of shape
                       num_layers x 2 (for c, h) x batch_size x in_size
    :param labels: placeholder for labels
    :param learning_rate: scalar placeholder for learning rate
    :return: train, loss and next_state nodes
    """
    # Indices of c and h states.
    c_i = 0
    h_i = 1
    # Empty list which will hold network outputs.
    outs = []
    # Unpack the rnn state.
    state = [
        (prev_state[i, c_i], prev_state[i, h_i])
        for i in range(num_layers)
    ]
    # Roll out the graph.
    for i in range(roll_out):
        with tf.variable_scope("", reuse=(i != 0)):
            out, state = cell(in_data[i], state)
        outs.append(out)
    # Apply output transformation.
    final_outs = tf.matmul(tf.concat(outs, axis=0), out_weights) + out_biases
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=final_outs,
            labels=tf.reshape(labels, [-1, num_classes])
        )
    )
    trainer = tf.train.AdagradOptimizer(learning_rate)
    # create training node with gradient clipping applied to it.
    grads = trainer.compute_gradients(loss)
    clipped_grads = [
        (tf.clip_by_value(grad, GRAD_MIN, GRAD_MAX), var)
        for grad, var in grads
    ]
    train = trainer.apply_gradients(clipped_grads)
    return train, loss, state


def load_set(set_dir_path):
    """
    Load data matrices and label vectors.
    :param set_dir_path: path to rnn set directory
    :return: list of matrices and labels
    """
    # Paths to data and labels directories which are inside of the set
    # directory.
    data_dir_path = "data"
    labels_dir_path = "labels"
    data_mats = []
    labels_mats = []
    for filename in listdir(set_dir_path + "/" + data_dir_path):
        # Read next company data and labels and save it in lists.
        data_mats.append(
            np.load(set_dir_path + "/" + data_dir_path + "/" + filename)
        )
        labels_mats.append(
            np.load(set_dir_path + "/" + labels_dir_path + "/" + filename)
        )
    return data_mats, labels_mats


def main():
    """
    Main function of the script.
    :return: -
    """
    # Number of inner states of lstm.
    num_states = 2
    # Path of directory which contains rnn data.
    set_dir_path = "../../data/rnn_set"
    # Load set.
    data_mats, labels_mats = load_set(set_dir_path)
    num_companies = len(data_mats)
    # RNN placeholders.
    in_data = tf.placeholder(
        dtype=tf.float32, shape=[ROLL_OUT, None, IN_SIZE]
    )
    previous_state = tf.placeholder(
        dtype=tf.float32, shape=[NUM_LAYERS, num_states, None, HIDDEN_SIZE]
    )
    labels = tf.placeholder(
        dtype=tf.float32, shape=[ROLL_OUT, None, NUM_CLASSES]
    )
    learning_rate = tf.placeholder(
        dtype=tf.float32, shape=[]
    )
    # Create the cell and output transformation.
    cell, out_weights, out_biases = create_cell(
        HIDDEN_SIZE, NUM_CLASSES, NUM_LAYERS
    )
    # Create training graph.
    train, loss, next_state = create_train_graph(
        cell, out_weights, out_biases, NUM_LAYERS, NUM_CLASSES, ROLL_OUT,
        in_data, previous_state, labels, learning_rate
    )
    # Create holders for intermediate state and zero state.
    zero_state = np.zeros([NUM_LAYERS, num_states, BATCH_SIZE, HIDDEN_SIZE])
    current_state = np.copy(zero_state)
    # Create and initialize session.
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    # Vector of indices of currently used companies and current indices in
    # their sets.
    no_company_index = -1.0
    companies_indices = np.random.permutation(num_companies)[:BATCH_SIZE]
    current_indices = np.zeros([BATCH_SIZE], dtype=np.int32)
    # Initialize starting loss.
    smooth_loss = -np.log(1.0 / NUM_CLASSES)
    keep_multi = 0.999
    # Container for one run data batch.
    data_batch = np.zeros([ROLL_OUT, BATCH_SIZE, IN_SIZE])
    labels_batch = np.zeros([ROLL_OUT, BATCH_SIZE, NUM_CLASSES])
    for i in range(MAX_ITERATIONS):
        # Keep selecting companies untill all of them in the companies indices
        # set have sufficient data to perform one train pass.
        while True:
            # Reset indices of companies whose current_i went over their data
            # size.
            companies_to_reset = companies_indices == no_company_index
            companies_indices[companies_to_reset] = np.random.permutation(
                num_companies
            )[:np.sum(companies_to_reset)]
            current_indices[companies_to_reset] = 0
            # Check which companies have still sufficient data to perform one
            # train pass. If they don't set their indices to no_company_index.
            for j in range(companies_indices.shape[0]):
                company_i = companies_indices[j]
                current_i = current_indices[j]
                if current_i + 2 * ROLL_OUT >= data_mats[company_i].shape[0]:
                    companies_indices[j] = no_company_index
                    # Set state of this company to 0.0. Reset state on all
                    # layers, both states and whole vectors for batch index
                    # of j.
                    current_state[:, :, j, :] = 0.0
            # Break if all companies have sufficient data.
            if no_company_index not in companies_indices:
                break
        # Update current indices of selected companies.
        for j in range(companies_indices.shape[0]):
            current_indices[j] += ROLL_OUT
        # Create data batch.
        for j in range(ROLL_OUT):
            for k in range(BATCH_SIZE):
                # Retrieve input vector and label vector from currently
                # processed k-th company with index j + current process
                # index for this company.
                data_batch[j, k] = data_mats[
                    companies_indices[k]
                ][j + current_indices[k]]
                labels_batch[j, k] = labels_mats[
                    companies_indices[k]
                ][j + current_indices[k]]
        # Perform train run.
        _, loss_value, next_state_tuple = session.run(
            [train, loss, next_state],
            feed_dict={
                in_data: data_batch,
                previous_state: current_state,
                labels: labels_batch,
                learning_rate: START_LEARNING_RATE
            }
        )
        # Convert net state from tuples to matrix.
        for j in range(NUM_LAYERS):
            current_state[j] = np.array(next_state_tuple[j])
        # Update smooth loss and print current state.
        smooth_loss = keep_multi * smooth_loss + (1 - keep_multi) * loss_value
        print(
            "Iteration {}:\n\tSmooth loss: {}\n\tLoss: {}".format(
                i, smooth_loss, loss_value
            )
        )


if __name__ == "__main__":
    main()
