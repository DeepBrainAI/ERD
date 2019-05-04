import tensorflow as tf
from tensorflow.contrib import rnn


class RL_GRU2:
    def __init__(self, input_dim, hidden_dim, max_seq_len, max_word_len, class_num, action_num):
        self.input_x = tf.placeholder(tf.float32, [None, max_seq_len, max_word_len, input_dim], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, class_num], name="input_y")
        self.x_len = tf.placeholder(tf.int32, [None], name="x_len")
        self.init_states = tf.placeholder(tf.float32, [None, hidden_dim], name="topics")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.rl_state = tf.placeholder(tf.float32, [None, hidden_dim], name="rl_states")
        self.rl_input = tf.placeholder(tf.float32, [None, max_word_len, input_dim], name="rl_input")
        self.action = tf.placeholder(tf.float32, [None, action_num], name="action")
        self.reward = tf.placeholder(tf.float32, [None], name="reward")

        output_dim = hidden_dim

        # shared pooling layer
        self.w_t = tf.Variable(tf.random_uniform([input_dim, output_dim], -1.0, 1.0), name="w_t")
        self.b_t = tf.Variable(tf.constant(0.01, shape=[output_dim]), name="b_t")
        pooled_input_x = self.shared_pooling_layer(self.input_x, input_dim, max_seq_len, max_word_len, output_dim)
        pooled_rl_input = self.shared_pooling_layer(self.rl_input, input_dim, 1, max_word_len, output_dim)
        pooled_rl_input = tf.reshape(pooled_rl_input, [-1, output_dim])

        # dropout layer
        pooled_input_x_dp = tf.nn.dropout(pooled_input_x, self.dropout_keep_prob)

        # df model
        df_cell = rnn.GRUCell(output_dim)
        df_cell = rnn.DropoutWrapper(df_cell, output_keep_prob=self.dropout_keep_prob)

        w_tp = tf.constant(0.0, shape=[hidden_dim, output_dim], name="w_tp")
        self.df_state = tf.matmul(self.init_states, w_tp, name="df_state")

        df_outputs, df_last_state = tf.nn.dynamic_rnn(df_cell, pooled_input_x_dp, self.x_len, initial_state=self.df_state, dtype=tf.float32)
        l2_loss = tf.constant(0.0)

        w_ps = tf.Variable(tf.truncated_normal([output_dim, class_num], stddev=0.1))
        b_ps = tf.Variable(tf.constant(0.01, shape=[class_num]))
        l2_loss += tf.nn.l2_loss(w_ps)
        l2_loss += tf.nn.l2_loss(b_ps)

        self.pre_scores = tf.nn.xw_plus_b(df_last_state, w_ps, b_ps, name="p_scores")
        self.predictions = tf.argmax(self.pre_scores, 1, name="predictions")

        r_outputs = tf.reshape(df_outputs, [-1, output_dim])
        scores_seq = tf.nn.softmax(tf.nn.xw_plus_b(r_outputs, w_ps, b_ps))
        self.out_seq = tf.reshape(scores_seq, [-1, max_seq_len, class_num], name="out_seq")

        df_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.pre_scores, labels=self.input_y)
        self.loss = tf.reduce_mean(df_losses) + 0.1 * l2_loss

        correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # rl model
        self.rl_output, self.rl_new_state = df_cell(pooled_rl_input, self.rl_state)

        w_ss1 = tf.Variable(tf.truncated_normal([output_dim, 64], stddev=0.01))
        b_ss1 = tf.Variable(tf.constant(0.01, shape=[64]))
        rl_h1 = tf.nn.relu(tf.nn.xw_plus_b(self.rl_state, w_ss1, b_ss1))

        w_ss2 = tf.Variable(tf.truncated_normal([64, action_num], stddev=0.01))
        b_ss2 = tf.Variable(tf.constant(0.01, shape=[action_num]))

        self.stopScore = tf.nn.xw_plus_b(rl_h1, w_ss2, b_ss2, name="stopScore")

        self.isStop = tf.argmax(self.stopScore, 1, name="isStop")

        out_action = tf.reduce_sum(tf.multiply(self.stopScore, self.action), reduction_indices=1)
        self.rl_cost = tf.reduce_mean(tf.square(self.reward - out_action), name="rl_cost")

    def shared_pooling_layer(self, inputs, input_dim, max_seq_len, max_word_len, output_dim):
        t_inputs = tf.reshape(inputs, [-1, input_dim])
        t_h = tf.nn.xw_plus_b(t_inputs, self.w_t, self.b_t)
        t_h = tf.reshape(t_h, [-1, max_word_len, output_dim])
        t_h_expended = tf.expand_dims(t_h, -1)
        pooled = tf.nn.max_pool(
            t_h_expended,
            ksize=[1, max_word_len, 1, 1],
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="max_pool"
        )
        outs = tf.reshape(pooled, [-1, max_seq_len, output_dim])
        return outs

    def pooling_layer(self, inputs, input_dim, max_seq_len, max_word_len, output_dim):
        t_inputs = tf.reshape(inputs, [-1, input_dim])
        w = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
        b = tf.Variable(tf.constant(0.01, shape=[output_dim]))

        h = tf.nn.xw_plus_b(t_inputs, w, b)
        hs = tf.reshape(h, [-1, max_word_len, output_dim])

        inputs_expended = tf.expand_dims(hs, -1)

        pooled = tf.nn.max_pool(
            inputs_expended,
            ksize=[1, max_word_len, 1, 1],
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="max_pool"
        )
        cnn_outs = tf.reshape(pooled, [-1, max_seq_len, output_dim])
        return cnn_outs
