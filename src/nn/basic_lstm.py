import numpy as np
import tensorflow as tf

from src.nn.config import LstmConfig as config
from src.nn.config import LstmConst as const


class Model(object):
    def __init__(self, input, output, train, loss, learning_rate):
        self.input = input
        self.output = output
        self.train = train
        self.loss = loss
        self.learning_rate = learning_rate


def create_graph(input, labels, learning_rate):
    lstm = tf.contrib.rnn.BasicLSTMCell(
        config.HIDDEN_SIZE, forget_bias=config.FORGET_BIAS, state_is_tuple=True)

    state = lstm.zero_state(config.UNFOLD_BATCH_SIZE, tf.float32)
    outputs = []

    for step in range(config.UNFOLD_BATCH_SIZE):
        with tf.variable_scope("", reuse=(step != 0)):
            out, state = lstm(input, state)
        outputs.append(out)

    W = tf.get_variable(
        "W", shape=[config.HIDDEN_SIZE, config.NUM_OF_LABELS],
        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(
        "b", shape=[config.NUM_OF_LABELS, ],
        initializer=tf.constant_initializer())

    output = tf.matmul(tf.concat(outputs, axis=0), W) + b

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=output, labels=labels))

    train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return Model(input, output, train, loss, learning_rate)


def train_network(run_index, set_index, mode, lr):
    # TODO preparing dataset

    learning_rate = tf.placeholder(tf.float32, shape=[])
    input = tf.placeholder(tf.float32, shape=[None, config.IN_SIZE])
    labels = tf.placeholder(dtype=tf.float32, shape=[None, config.NUM_OF_LABELS])

    model = create_graph(input, labels, learning_rate)

    # TODO data retrieval
    batch_data = np.random.rand(config.UNFOLD_BATCH_SIZE, config.IN_SIZE)
    batch_labels = np.random.rand(config.UNFOLD_BATCH_SIZE * config.UNFOLD_BATCH_SIZE,
            config.NUM_OF_LABELS)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    for step in range(config.MAX_ITERATIONS):
        o, l, _ = session.run([model.output, model.loss, model.train],
            feed_dict={input: batch_data, labels: batch_labels,
            learning_rate: lr})
        print(step, " loss = ", l)


def main():
    # TODO
    train_network(const.RUN_INDEX, const.SET_INDEX, const.MODE, config.LEARNING_RATE)


if __name__ == "__main__":
    main()
