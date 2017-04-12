# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


# Actor-Critic Network Base Class
# (Policy network and Value network)
class GameACNetwork(object):
    def __init__(self,
                 action_size,
                 thread_index,  # -1 for global
                 learning_rate_input,
                 device="/cpu:0"):
        self._action_size = action_size
        self._thread_index = thread_index
        self._device = device
        self.learning_rate_input = learning_rate_input

    def prepare_loss(self, entropy_beta):
        with tf.device(self._device):
            # taken action (input for policy)
            self.a = tf.placeholder("float", [None, self._action_size])

            # temporary difference (R-V) (input for policy)
            self.td = tf.placeholder("float", [None])

            # avoid NaN with clipping when value in pi becomes zero
            log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0))

            # policy entropy
            entropy = -tf.reduce_sum(self.pi * log_pi, reduction_indices=1)

            # policy loss (output)  (Adding minus, because the original paper's objective function is for gradient ascent, but we use gradient descent optimizer.)
            policy_loss = - tf.reduce_sum(
                tf.reduce_sum(tf.multiply(log_pi, self.a), reduction_indices=1) * self.td + entropy * entropy_beta)

            # R (input for value)
            self.r = tf.placeholder("float", [None])

            # value loss (output)
            # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
            value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)

            # gradient of policy and value are summed up
            self.total_loss = policy_loss + value_loss


    def start_train(self):
        pass  # Used to save any data that might be needed at the end

    def run_policy_and_value(self, sess, s_t):
        raise NotImplementedError()

    def run_policy(self, sess, s_t):
        raise NotImplementedError()

    def run_value(self, sess, s_t):
        raise NotImplementedError()

    def get_vars(self):
        raise NotImplementedError()

    def apply_gradients(self, sess, apply_gradients, batch_si, batch_a, batch_td, batch_R, cur_learn_rate):
        raise NotImplementedError()

    def sync_from(self, src_network, name=None):
        src_vars = src_network.get_vars()
        dst_vars = self.get_vars()

        sync_ops = []

        with tf.device(self._device):
            with tf.name_scope(name, "GameACNetwork", []) as name:
                for (src_var, dst_var) in zip(src_vars, dst_vars):
                    sync_op = tf.assign(dst_var, src_var)
                    sync_ops.append(sync_op)

                return tf.group(*sync_ops, name=name)


    # weight initialization based on muupan's code
    # https://github.com/muupan/async-rl/blob/master/a3c_ale.py
    def _fc_variable(self, weight_shape):
        input_channels = weight_shape[0]
        output_channels = weight_shape[1]
        d = 1.0 / np.sqrt(input_channels)
        bias_shape = [output_channels]
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
        bias = tf.Variable(tf.random_uniform(bias_shape, minval=-d, maxval=d))
        return weight, bias

    def _conv_variable(self, weight_shape):
        depth = weight_shape[0]
        w = weight_shape[1]
        h = weight_shape[2]
        input_channels = weight_shape[3]
        output_channels = weight_shape[4]
        d = 1.0 / np.sqrt(input_channels * depth * w * h)
        bias_shape = [output_channels]
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
        bias = tf.Variable(tf.random_uniform(bias_shape, minval=-d, maxval=d))
        return weight, bias

    def _conv2d(self, x, W, stride):
        return tf.nn.conv3d(x, W, strides=[1, stride, stride, stride, 1], padding="VALID")


# Actor-Critic LSTM Network
class GameACLSTMNetwork(GameACNetwork):
    def __init__(self,
                 action_size,
                 thread_index,  # -1 for global
                 learning_rate_input,
                 device="/cpu:0"):
        GameACNetwork.__init__(self, action_size, thread_index, learning_rate_input, device)

        scope_name = "net_" + str(self._thread_index)
        with tf.device(self._device), tf.variable_scope(scope_name) as scope:
            self.W_conv1, self.b_conv1 = self._conv_variable([2, 8, 8, 3, 16])  # stride=4
            self.W_conv2, self.b_conv2 = self._conv_variable([1, 4, 4, 16, 32])  # stride=2

            self.W_fc1, self.b_fc1 = self._fc_variable([2304, 512])

            # lstm
            self.lstm = tf.contrib.rnn.BasicLSTMCell(512, state_is_tuple=True)

            # weight for policy output layer
            self.W_fc2, self.b_fc2 = self._fc_variable([512, action_size])

            # weight for value output layer
            self.W_fc3, self.b_fc3 = self._fc_variable([512, 1])

            # feature layers for enemy detection
            self.W_fc4, self.b_fc4 = self._fc_variable([2304, 512])
            self.W_fc5, self.b_fc5 = self._fc_variable([512, 2])

            # state (input)
            self.s = tf.placeholder("float", [None, 4, 108, 60, 3])

            h_conv1 = tf.nn.relu(self._conv2d(self.s, self.W_conv1, 4) + self.b_conv1)
            h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)

            h_conv2_flat = tf.reshape(h_conv2, [-1,  2304])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)
            # h_fc1 shape=(5,512)

            h_fc1_reshaped = tf.reshape(h_fc1, [1, -1, 512])
            # h_fc_reshaped = (1,5,512)

            # place holder for LSTM unrolling time step size.
            self.step_size = tf.placeholder(tf.float32, [1])

            self.initial_lstm_state0 = tf.placeholder(tf.float32, [1, 512])
            self.initial_lstm_state1 = tf.placeholder(tf.float32, [1, 512])
            self.initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self.initial_lstm_state0,
                                                                    self.initial_lstm_state1)

            # Unrolling LSTM up to LOCAL_T_MAX time steps. (= 5time steps.)
            # When episode terminates unrolling time steps becomes less than LOCAL_TIME_STEP.
            # Unrolling step size is applied via self.step_size placeholder.
            # When forward propagating, step_size is 1.
            # (time_major = False, so output shape is [batch_size, max_time, cell.output_size])
            lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm,
                                                              h_fc1_reshaped,
                                                              initial_state=self.initial_lstm_state,
                                                              sequence_length=self.step_size,
                                                              time_major=False,
                                                              scope=scope)

            # lstm_outputs: (1,5,256) for back prop, (1,1,512) for forward prop.

            lstm_outputs = tf.reshape(lstm_outputs, [-1, 512])

            # Fully connected feature layer
            h_fc4 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc4) + self.b_fc4)
            self.features = tf.nn.relu(tf.matmul(h_fc4, self.W_fc5) + self.b_fc5)

            # policy (output)
            self.pi = tf.nn.softmax(tf.matmul(lstm_outputs, self.W_fc2) + self.b_fc2)

            # value (output)
            v_ = tf.matmul(lstm_outputs, self.W_fc3) + self.b_fc3
            self.v = tf.reshape(v_, [-1])

            scope.reuse_variables()
            self.W_lstm = tf.get_variable("basic_lstm_cell/weights")
            self.b_lstm = tf.get_variable("basic_lstm_cell/biases")

            self.reset_state()
            self.start_train()

    def reset_state(self):
        self.lstm_state_out = tf.contrib.rnn.LSTMStateTuple(np.zeros([1, 512]),
                                                            np.zeros([1, 512]))

    def start_train(self):
        self.start_lstm_state = self.lstm_state_out

    def run_policy_and_value(self, sess, s_t):
        # This run_policy_and_value() is used when forward propagating.
        # so the step size is 1.
        pi_out, v_out, self.lstm_state_out = sess.run([self.pi, self.v, self.lstm_state],
                                                      feed_dict={self.s: [s_t],
                                                                 self.initial_lstm_state0: self.lstm_state_out[0],
                                                                 self.initial_lstm_state1: self.lstm_state_out[1],
                                                                 self.step_size: [1]})
        # pi_out: (1,3), v_out: (1)
        return (pi_out[0], v_out[0])

    def run_policy(self, sess, s_t):
        # This run_policy() is used for displaying the result with display tool.
        pi_out, self.lstm_state_out = sess.run([self.pi, self.lstm_state],
                                               feed_dict={self.s: [s_t],
                                                          self.initial_lstm_state0: self.lstm_state_out[0],
                                                          self.initial_lstm_state1: self.lstm_state_out[1],
                                                          self.step_size: [1]})
        return pi_out[0]

    def run_value(self, sess, s_t):
        # This run_value() is used for calculating V for bootstrapping at the
        # end of LOCAL_T_MAX time step sequence.
        # When next sequcen starts, V will be calculated again with the same state using updated network weights,
        # so we don't update LSTM state here.
        prev_lstm_state_out = self.lstm_state_out
        v_out, _ = sess.run([self.v, self.lstm_state],
                            feed_dict={self.s: [s_t],
                                       self.initial_lstm_state0: self.lstm_state_out[0],
                                       self.initial_lstm_state1: self.lstm_state_out[1],
                                       self.step_size: [1]})

        # roll back lstm state
        self.lstm_state_out = prev_lstm_state_out
        return v_out[0]

    def run_feature_detection(self, sess, s_t):
        # Returns list of [prediction as probability]
        detection = sess.run(self.features, feed_dict={
            self.s: [s_t]
        })
        return detection[0]

    def prepare_loss(self, entropy_beta):
        super(GameACLSTMNetwork, self).prepare_loss(entropy_beta)
        # input for feature
        # self.feature = tf.placeholder("float", [None, 1])
        self.actual_feature = tf.placeholder(tf.int32, [None])
        self.feature_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.features, labels=self.actual_feature)

    def get_vars(self):
        return [self.W_conv1, self.b_conv1,
                self.W_conv2, self.b_conv2,
                self.W_fc1, self.b_fc1,
                self.W_lstm, self.b_lstm,
                self.W_fc2, self.b_fc2,
                self.W_fc3, self.b_fc3]

    def get_feature_vars(self):
        return [self.W_conv1, self.b_conv1,
                self.W_conv2, self.b_conv2,
                self.W_fc4, self.b_fc4,
                self.W_fc5, self.b_fc5]

    def apply_gradients(self, sess, apply_gradients, batch_si, batch_a, batch_td, batch_R, cur_learn_rate):
        batch_si.reverse()
        batch_a.reverse()
        batch_td.reverse()
        batch_R.reverse()

        sess.run(apply_gradients,
                 feed_dict={
                     self.s: batch_si,
                     self.a: batch_a,
                     self.td: batch_td,
                     self.r: batch_R,
                     self.initial_lstm_state: self.start_lstm_state,
                     self.step_size: [len(batch_a)],
                     self.learning_rate_input: cur_learn_rate})

    def apply_feature_gradient(self, sess, apply_gradients, batch_si, batch_pred, batch_real, cur_learn_rate):
        sess.run(apply_gradients,
                 feed_dict={
                 self.s: batch_si,
                 self.features: batch_pred,
                 self.actual_feature: batch_real,
                 self.learning_rate_input: cur_learn_rate})

    def sync_from(self, src_network, name=None):
        src_vars = src_network.get_vars()
        src_feat_vars = src_network.get_feature_vars()
        dst_vars = self.get_vars()
        dst_feat_vars = self.get_feature_vars()

        sync_ops = []

        with tf.device(self._device):
            with tf.name_scope(name, "GameACNetwork", []) as name:
                for (src_var, dst_var) in zip(src_vars, dst_vars):
                    sync_op = tf.assign(dst_var, src_var)
                    sync_ops.append(sync_op)
                for (src_var, dst_var) in zip(src_feat_vars, dst_feat_vars):
                    sync_op = tf.assign(dst_var, src_var)
                    sync_ops.append(sync_op)

                return tf.group(*sync_ops, name=name)