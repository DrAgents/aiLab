import tensorflow as tf
from tensorflow.contrib import rnn
from annoy import AnnoyIndex
import utils

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
gpu_options.allow_growth = True
session_conf = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=True,
    gpu_options=gpu_options
)


class QaModel(object):
    """anan doctor qa model
    """

    def __init__(self, checkpoint_dir, tree_name, table_name, query_steps=12, lstm_size=500):

        self.query_steps = query_steps
        self.lstm_size = lstm_size
        self.t = self.load_ann_tree(tree_name)
        self.table_name = table_name

        graph = tf.Graph()
        with graph.as_default():
            self.sess = sess = tf.Session(config=session_conf)
            with sess.as_default():
                # inputs
                self.input_placeholder = tf.placeholder(tf.float32, shape=[1, None, 256])

                # outputs
                self.question_vec = self.encoder(self.input_placeholder)

                saver = tf.train.Saver(tf.trainable_variables())
                ckpt = tf.train.latest_checkpoint(checkpoint_dir)
                if ckpt:
                    saver.restore(sess, ckpt)
                else:
                    raise("Not find the checkpoint!")

    def encoder(self, place_holder):
        with tf.variable_scope("LSTM_scope"):
            q = bi_lstm(place_holder, self.lstm_size, self.query_steps)
            question_feat = tf.nn.tanh(max_pooling(q))
        return question_feat

    def load_ann_tree(self, tree_name):
        t = AnnoyIndex(1000, metric='euclidean')
        t.load(tree_name)
        return t

    def prediction(self, query, cursor, num=3, contain_query=False):
        q_enc = self.sess.run(self.question_vec,
                              {self.input_placeholder: [utils.sentence2vector(query, self.query_steps)]})
        sim = self.t.get_nns_by_vector(q_enc[0], num, search_k=100000, include_distances=True)
        predictions = []
        for i in range(num):
            cursor.execute("SELECT question, answer FROM {} WHERE id = ?".format(self.table_name), (sim[0][i], ))
            question, answer = cursor.fetchone()
            if contain_query:
                predictions.append((answer, sim[1][i], question))
            else:
                predictions.append((answer, sim[1][i]))
        return predictions



def bi_lstm(x, n_hidden, n_steps):
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, axis=1)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                 dtype=tf.float32)
    # output transformation to the original tensor type
    outputs = tf.stack(outputs, axis=1)
    return outputs

def max_pooling(lstm_out):
    height, width = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])  # (step, length of input for one step)

    # do max-pooling to change the (sequence_lenght) tensor to 1-lenght tensor
    lstm_out = tf.expand_dims(lstm_out, -1)
    output = tf.nn.max_pool(
        lstm_out,
        ksize=[1, height, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID')

    output = tf.reshape(output, [-1, width])
    return output