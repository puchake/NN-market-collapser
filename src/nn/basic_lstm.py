import tensorflow as tf
import numpy as np
from os import listdir
from matplotlib import pyplot as plt


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
TRAIN_BATCH_SIZE = 200
VALIDATION_BATCH_SIZE = 50
TEST_BATCH_SIZE = 50
MAX_ITERATIONS = 1000000
VALIDATION_INTERVAL = 100
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
    :return: train, loss, accuracy, predictions and next_state nodes
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
    predictions = tf.argmax(final_outs, axis=1)
    correct = tf.argmax(tf.reshape(labels, [-1, num_classes]), axis=1)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(predictions, correct), tf.float32)
    )
    trainer = tf.train.AdagradOptimizer(learning_rate)
    # create training node with gradient clipping applied to it.
    grads = trainer.compute_gradients(loss)
    clipped_grads = [
        (tf.clip_by_value(grad, GRAD_MIN, GRAD_MAX), var)
        for grad, var in grads
    ]
    train = trainer.apply_gradients(clipped_grads)
    return train, loss, accuracy, predictions, final_outs, state


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
    # Split data_mats and labels_mats for train, validation and test subsets.
    train_data_mats = []
    train_labels_mats = []
    validation_data_mats = []
    validation_labels_mats = []
    test_data_mats = []
    test_labels_mats = []
    # Desired validation and train set sizes and minimal required row count of
    # company matrix which will be added to one of these sets.
    validation_set_size = 50
    test_set_size = 50
    required_row_count = 4000
    for i in range(len(data_mats)):
        # If we still don't have enough matrices in validation set or test
        # set and if current matrix has required size, add it to one of these
        # sets.
        if len(validation_data_mats) < validation_set_size and \
                data_mats[i].shape[0] >= required_row_count:
            validation_data_mats.append(data_mats[i])
            validation_labels_mats.append(labels_mats[i])
        elif len(test_data_mats) < test_set_size and \
                data_mats[i].shape[0] >= required_row_count:
            test_data_mats.append(data_mats[i])
            test_labels_mats.append(labels_mats[i])
        # Else add current matrix with its labels to train set.
        else:
            train_data_mats.append(data_mats[i])
            train_labels_mats.append(labels_mats[i])
    return train_data_mats, train_labels_mats, \
           validation_data_mats, validation_labels_mats, \
           test_data_mats, test_labels_mats


def visualize_validation(i, j, data_mat, labels_mat, rnn_outs):
    """
    Show example rnn result on scatter plot.
    :param i: global iteration
    :param j: validation iteration
    :param data_mat: matrix of rnn input data for example company
    :param labels_mat: matrix of correct labels for this company
    :param rnn_outs: scores output by rnn
    :return: -
    """
    # Pattern of plot filename.
    plot_path = "../../data/rnn_log/images/plot_{:06d}_{:04d}.png"
    # Plotted figure size.
    figure_size = (15, 7.5)
    # Size of markers for class and score.
    score_size = 200
    class_size = 50
    # Indices of plotted values.
    close_i = 0
    avg_25_i = 2
    # Derive classes from labels.
    classes = np.argmax(labels_mat, axis=1)
    # Count scores output by rnn.
    scores = np.exp(rnn_outs)
    scores = np.divide(scores.T, np.sum(scores, axis=1)).T
    scores = scores[:, 1] - scores[:, 0]
    x = np.arange(data_mat.shape[0])
    # Plot scores classes and avg_25 and save it to file.
    plt.figure(1, figsize=figure_size)
    plt.clf()
    plt.plot(x, data_mat[:, avg_25_i], 'b')
    # Plot scores with color scale.
    plt.scatter(
        x, data_mat[:, close_i], c=scores, cmap="seismic",
        edgecolors='black', s=score_size, vmin=-1.0, vmax=1.0
    )
    # Plot real classes.
    plt.scatter(
        x, data_mat[:, close_i], c=classes, cmap="jet",
        edgecolors='black', s=class_size
    )
    # Save the figure.
    plt.savefig(plot_path.format(i, j))
    plt.close()


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
    train_data_mats, train_labels_mats, \
    validation_data_mats, validation_labels_mats, \
    test_data_mats, test_labels_mats = load_set(set_dir_path)
    num_train_companies = len(train_data_mats)
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
    train, loss, accuracy, predictions, final_outs, next_state = \
        create_train_graph(
            cell, out_weights, out_biases, NUM_LAYERS, NUM_CLASSES, ROLL_OUT,
            in_data, previous_state, labels, learning_rate
        )
    # Create holders for intermediate state and zero state.
    train_current_state = np.zeros(
        [NUM_LAYERS, num_states, TRAIN_BATCH_SIZE, HIDDEN_SIZE]
    )
    validation_current_state = np.zeros(
        [NUM_LAYERS, num_states, VALIDATION_BATCH_SIZE, HIDDEN_SIZE]
    )
    test_current_state = np.zeros(
        [NUM_LAYERS, num_states, TEST_BATCH_SIZE, HIDDEN_SIZE]
    )
    # Create and initialize session.
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    # Vector of indices of currently used companies in training and current
    # indices in their sets.
    no_company_index = -1.0
    train_companies_indices = np.random.permutation(
        num_train_companies
    )[:TRAIN_BATCH_SIZE]
    train_current_indices = np.zeros([TRAIN_BATCH_SIZE], dtype=np.int32)
    # Initialize starting loss.
    smooth_loss = -np.log(1.0 / NUM_CLASSES)
    keep_multi = 0.99
    # Container for one data batch of either train, validation or test run.
    batch_size_limit = np.max(
        [TRAIN_BATCH_SIZE, VALIDATION_BATCH_SIZE, TEST_BATCH_SIZE]
    )
    data_batch = np.zeros([ROLL_OUT, batch_size_limit, IN_SIZE])
    labels_batch = np.zeros([ROLL_OUT, batch_size_limit, NUM_CLASSES])
    # Minimal pause to handle plot events.
    plot_pause = 0.05
    for i in range(MAX_ITERATIONS):
        # Keep selecting companies until all of them in the companies indices
        # set have sufficient data to perform one train pass.
        while True:
            # Reset indices of companies whose current_i went over their data
            # size.
            companies_to_reset = train_companies_indices == no_company_index
            train_companies_indices[companies_to_reset] = \
                np.random.permutation(
                    num_train_companies
                )[:np.sum(companies_to_reset)]
            train_current_indices[companies_to_reset] = 0
            # Check which companies have still sufficient data to perform one
            # train pass. If they don't set their indices to no_company_index.
            for j in range(train_companies_indices.shape[0]):
                company_i = train_companies_indices[j]
                current_i = train_current_indices[j]
                if current_i + 2 * ROLL_OUT >= \
                        train_data_mats[company_i].shape[0]:
                    train_companies_indices[j] = no_company_index
                    # Set state of this company to 0.0. Reset state on all
                    # layers, both states and whole vectors for batch index
                    # of j.
                    train_current_state[:, :, j, :] = 0.0
            # Break if all companies have sufficient data.
            if no_company_index not in train_companies_indices:
                break
        # Update current indices of selected companies.
        for j in range(train_companies_indices.shape[0]):
            train_current_indices[j] += ROLL_OUT
        # Create data batch.
        for j in range(ROLL_OUT):
            for k in range(TRAIN_BATCH_SIZE):
                # Retrieve input vector and label vector from currently
                # processed k-th company with index j + current process
                # index for this company.
                data_batch[j, k] = train_data_mats[
                    train_companies_indices[k]
                ][j + train_current_indices[k]]
                labels_batch[j, k] = train_labels_mats[
                    train_companies_indices[k]
                ][j + train_current_indices[k]]
        # Perform train run.
        _, loss_value, accuracy_value, next_state_tuple = session.run(
            [train, loss, accuracy, next_state],
            feed_dict={
                in_data: data_batch[:, :TRAIN_BATCH_SIZE],
                previous_state: train_current_state,
                labels: labels_batch[:, :TRAIN_BATCH_SIZE],
                learning_rate: START_LEARNING_RATE
            }
        )
        # Convert net state from tuples to matrix.
        for j in range(NUM_LAYERS):
            train_current_state[j] = np.array(next_state_tuple[j])
        # Update smooth loss and print current state.
        smooth_loss = keep_multi * smooth_loss + (1 - keep_multi) * loss_value
        print(
            ("Iteration {}:\n\tSmooth loss: {}\n\tLoss: {}" +
             "\n\tAccuracy: {}").format(
                i, smooth_loss, loss_value, accuracy_value
            )
        )
        # Perform validation with set interval.
        if i % VALIDATION_INTERVAL == 0:
            print("Iteration {} - Validation:".format(i))
            # Zero current companies indices.
            validation_current_indices = np.zeros(
                [len(validation_data_mats)], dtype=np.int32
            )
            validation_correct_indices = validation_current_indices.copy()
            incorrect_validation_index = -1.0
            visualize_company_i = 0
            # Zero validation state.
            validation_current_state[:, :, :, :] = 0.0
            # Arrays for accuracies and losses.
            accuracies = []
            losses = []
            # Validation loop index with visualization interval.
            validation_i = 0
            visualization_interval = 5
            # Perform validation runs until one company runs out of data.
            while incorrect_validation_index not in validation_correct_indices:
                # Create data batch.
                for j in range(ROLL_OUT):
                    for k in range(len(validation_data_mats)):
                        # Retrieve input vector and label vector from currently
                        # processed k-th company with index j + current process
                        # index for this company.
                        data_batch[j, k] = validation_data_mats[
                            k
                        ][j + validation_current_indices[k]]
                        labels_batch[j, k] = validation_labels_mats[
                            k
                        ][j + validation_current_indices[k]]
                # Perform train run.
                loss_value, accuracy_value, next_outs, next_state_tuple = \
                    session.run(
                        [loss, accuracy, final_outs, next_state],
                        feed_dict={
                            in_data: data_batch[:, :VALIDATION_BATCH_SIZE],
                            previous_state: validation_current_state,
                            labels: labels_batch[:, :VALIDATION_BATCH_SIZE],
                            learning_rate: START_LEARNING_RATE
                        }
                    )
                # Record accuracy and loss.
                accuracies.append(accuracy_value)
                losses.append(loss_value)
                # Convert net state from tuples to matrix.
                for j in range(NUM_LAYERS):
                    validation_current_state[j] = np.array(next_state_tuple[j])
                # Update current indices of selected companies.
                for j in range(validation_current_indices.shape[0]):
                    validation_current_indices[j] += ROLL_OUT
                    # If there is no sufficient amount of data in currently
                    # processed company set its correct index to incorrect.
                    if validation_current_indices[j] + ROLL_OUT >= \
                            validation_data_mats[j].shape[0]:
                        validation_correct_indices[j] = \
                            incorrect_validation_index
                # Visualize validation with interval.
                if validation_i % visualization_interval == 0:
                    visualize_validation(
                        i,
                        validation_i,
                        data_batch[:, visualize_company_i],
                        labels_batch[:, visualize_company_i],
                        next_outs[::VALIDATION_BATCH_SIZE]
                    )
                validation_i += 1
            # Output mean loss and mean accuracy for validation.
            print(
                "\tMean loss: {}\n\tMean accuracy: {}".format(
                    np.mean(losses), np.mean(accuracies)
                )
            )


if __name__ == "__main__":
    main()
