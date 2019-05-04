import tensorflow as tf


tf.flags.DEFINE_string("w2v_file_path", "w2v_300", "w2v file")
tf.flags.DEFINE_string("data_file_path", "D:\\pheme-rnr-dataset", "data_file")

tf.flags.DEFINE_integer("post_fn", 2, "Fixed Number of Posts")
tf.flags.DEFINE_integer("time_limit", 48, "Posts Time Limitation (The Posts in 48 Hours)")

tf.flags.DEFINE_integer("batch_size", 50, "Batch size (default: 50)")
tf.flags.DEFINE_integer("hidden_dim", 100, "Dimensionality of hidden states (default: 100)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of word embedding (default: 300)")
tf.flags.DEFINE_integer("max_seq_len", 100, "Max length of sequence (default: 300)")
tf.flags.DEFINE_integer("max_sent_len", 100, "Max length of sentence (default: 300)")

tf.flags.DEFINE_integer("class_num", 2, "#Class (Non-Rumor, Rumor)")
tf.flags.DEFINE_integer("action_num", 2, "#Action (Continue, Stop)")

# RL parameters;
tf.flags.DEFINE_float("random_rate", 0.01, "RL Random Action Rate")
tf.flags.DEFINE_integer("OBSERVE", 1000, "OBSERVE BEFORE TRAIN")
tf.flags.DEFINE_integer("max_memory", 80000, "Max memory size")
tf.flags.DEFINE_float("reward_rate", 0.2, "reward rate")

FLAGS = tf.flags.FLAGS
