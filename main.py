# coding: utf-8
from collections import deque
from model import RL_GRU2
from dataUtils import *

tf.logging.set_verbosity(tf.logging.ERROR)


def df_train(sess, mm, t_acc, t_steps, new_data_len=[]):
    sum_loss = 0.0
    sum_acc = 0.0
    ret_acc = 0.0
    init_states = np.zeros([FLAGS.batch_size, FLAGS.hidden_dim], dtype=np.float32)

    for i in range(t_steps):
        if len(new_data_len) > 0:
            x, x_len, y = get_df_batch(i, new_data_len)
        else:
            x, x_len, y = get_df_batch(i)
        feed_dic = {mm.input_x: x, mm.x_len: x_len, mm.input_y: y, mm.init_states: init_states, mm.dropout_keep_prob: 0.5}
        _, step, loss, acc = sess.run([df_train_op, df_global_step, mm.loss, mm.accuracy], feed_dic)
        sum_loss += loss
        sum_acc += acc

        if i % 100 == 99:
            sum_loss = sum_loss / 100
            sum_acc = sum_acc / 100
            ret_acc = sum_acc
            print(get_curtime() + " Step: " + str(step) + " Training loss: " + str(sum_loss) + " accuracy: " + str(sum_acc))
            if sum_acc > t_acc:
                break
            sum_acc = 0.0
            sum_loss = 0.0

    print(get_curtime() + " Train df Model End.")
    return ret_acc


def rl_train(sess, mm, t_rw, t_steps):
    ids = np.array(range(FLAGS.batch_size), dtype=np.int32)
    seq_states = np.zeros([FLAGS.batch_size], dtype=np.int32)
    isStop = np.zeros([FLAGS.batch_size], dtype=np.int32)
    max_id = FLAGS.batch_size
    init_states = np.zeros([FLAGS.batch_size, FLAGS.hidden_dim], dtype=np.float32)
    state = sess.run(mm.df_state, feed_dict={mm.topics: init_states})

    D = deque()
    ssq = []
    print("in RL the begining")
    # get_new_len(sess, mm)
    if len(data_ID) % FLAGS.batch_size == 0:
        flags = len(data_ID) / FLAGS.batch_size
    else:
        flags = len(data_ID) / FLAGS.batch_size + 1
    for i in range(flags):
        x, x_len, y = get_df_batch(i)
        feed_dic = {mm.input_x: x, mm.x_len: x_len, mm.input_y: y, mm.topics: init_states, mm.dropout_keep_prob: 1.0}
        t_ssq = sess.run(mm.out_seq, feed_dic)
        if len(ssq) > 0:
            ssq = np.append(ssq, t_ssq, axis=0)
        else:
            ssq = t_ssq
    print(get_curtime() + " Now Start RL training ...")
    counter = 0
    sum_rw = 0.0
    while True:
        if counter > FLAGS.OBSERVE:
            sum_rw += np.mean(rw)
            if counter % 200 == 0:
                sum_rw = sum_rw / 2000
                print(get_curtime() + " Step: " + str(step) + " REWARD IS " + str(sum_rw))
                if sum_rw > t_rw:
                    print("Retch The Target Reward")
                    break
                if counter > t_steps:
                    print("Retch The Target Steps")
                    break
                sum_rw = 0.0
            s_state, s_x, s_isStop, s_rw = get_RL_Train_batch(D)
            feed_dic = {mm.rl_state: s_state, mm.rl_input: s_x, mm.action: s_isStop, mm.reward:s_rw, mm.dropout_keep_prob: 0.5}
            _, step = sess.run([rl_train_op, rl_global_step], feed_dic)

        x, y, ids, seq_states, max_id = get_rl_batch(ids, seq_states, isStop, max_id, 0, 3150)
        batch_dic = {mm.rl_state: state, mm.rl_input: x, mm.dropout_keep_prob: 1.0}
        isStop, mss, mNewState = sess.run([mm.isStop, mm.stopScore, mm.rl_new_state], batch_dic)

        for j in range(FLAGS.batch_size):
            if random.random() < FLAGS.random_rate:
                isStop[j] = np.argmax(np.random.rand(2))
            if seq_states[j] == data_len[ids[j]]:
                isStop[j] = 1
        # eval
        rw = get_reward(isStop, mss, ssq, ids, seq_states)

        for j in range(FLAGS.batch_size):
            D.append((state[j], x[j], isStop[j], rw[j]))
            if len(D) > FLAGS.max_memory:
                D.popleft()

        state = mNewState
        for j in range(FLAGS.batch_size):
            if isStop[j] == 1:
                init_states = np.zeros([FLAGS.batch_size, FLAGS.hidden_dim], dtype=np.float32)
                state[j] = sess.run(mm.df_state, feed_dict={mm.topics: init_states})

        counter += 1


def eval(sess, mm):
    start_ef = int(eval_flag / FLAGS.batch_size)
    end_ef = int(len(data_ID) / FLAGS.batch_size) + 1
    init_states = np.zeros([FLAGS.batch_size, FLAGS.hidden_dim], dtype=np.float32)

    counter = 0
    sum_acc = 0.0

    for i in range(start_ef, end_ef):
        x, x_len, y = get_df_batch(i)
        feed_dic = {mm.input_x: x, mm.x_len: x_len, mm.input_y: y, mm.init_states: init_states, mm.dropout_keep_prob: 1.0}
        _, step, loss, acc = sess.run([df_train_op, df_global_step, mm.loss, mm.accuracy], feed_dic)
        counter += 1
        sum_acc += acc

    print(sum_acc / counter)


if __name__ == "__main__":
    print(get_curtime() + " Loading data ...")
    load_data(FLAGS.data_file_path)
    print(get_curtime() + " Data loaded.")

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            # (self, input_dim, hidden_dim, max_seq_len, max_word_len, class_num, action_num):
            print(FLAGS.embedding_dim, FLAGS.hidden_dim, FLAGS.max_seq_len, FLAGS.max_sent_len, FLAGS.class_num, FLAGS.action_num)
            mm = RL_GRU2(FLAGS.embedding_dim, FLAGS.hidden_dim, FLAGS.max_seq_len,
                         FLAGS.max_sent_len, FLAGS.class_num, FLAGS.action_num)

            # df model
            df_global_step = tf.Variable(0, name="global_step", trainable=False)
            df_train_op = tf.train.AdamOptimizer(0.01).minimize(mm.loss, df_global_step)

            # rl model
            rl_global_step = tf.Variable(0, name="global_step", trainable=False)
            rl_train_op = tf.train.AdamOptimizer(0.001).minimize(mm.rl_cost, rl_global_step)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=4)

            sess.run(tf.global_variables_initializer())

            ckpt_dir = "df_saved"
            checkpoint = tf.train.get_checkpoint_state(ckpt_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print(checkpoint.model_checkpoint_path+" is restored.")
            else:
                df_train(sess, mm, 0.80, 2000)
                saver.save(sess, "df_saved/model")
                print("df_model "+" saved")

            for i in range(20):
                rl_train(sess, mm, 0.5, 50000)
                saver.save(sess, "rl_saved/model"+str(i))
                print("rl_model "+str(i)+" saved")
                new_len = get_new_len(sess, mm)
                acc = df_train(sess, mm, 0.9, 500, new_len)
                saver.save(sess, "df_saved/model"+str(i))
                print("df_model "+str(i)+" saved")
                if acc > 0.9:
                    break

    print("The End of My Program")
