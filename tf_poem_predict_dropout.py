import collections
import numpy as np
import tensorflow as tf

# -------------------------------数据预处理---------------------------#
import time

poetry_file = 'poetry.txt'

# 诗集
poetrys = []
with open(poetry_file, "r", encoding='utf-8', ) as f:
    for line in f:
        try:
            title, content = line.strip().split(':')
            content = content.replace(' ', '')
            if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
                continue
            if len(content) < 5 or len(content) > 79:
                continue
            content = '[' + content + ']'
            poetrys.append(content)
        except Exception as e:
            pass

# 按诗的字数排序
poetrys = sorted(poetrys, key=lambda line: len(line))
print('唐诗总数: ', len(poetrys))

# 统计每个字出现次数
all_words = []
for poetry in poetrys:
    all_words += [word for word in poetry]
counter = collections.Counter(all_words)
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
words, _ = zip(*count_pairs)

# 取前多少个常用字
words = words[:len(words)] + (' ',)
# 每个字映射为一个数字ID
word_num_map = dict(zip(words, range(len(words))))
# 把诗转换为向量形式，参考TensorFlow练习1
to_num = lambda word: word_num_map.get(word, len(words))
poetrys_vector = [list(map(to_num, poetry)) for poetry in poetrys]
# [[314, 3199, 367, 1556, 26, 179, 680, 0, 3199, 41, 506, 40, 151, 4, 98, 1],
# [339, 3, 133, 31, 302, 653, 512, 0, 37, 148, 294, 25, 54, 833, 3, 1, 965, 1315, 377, 1700, 562, 21, 37, 0, 2, 1253, 21, 36, 264, 877, 809, 1]
# ....]

batch_size = 1
n_chunk = len(poetrys_vector) // batch_size
x_batches = []
y_batches = []
for i in range(n_chunk):
    start_index = i * batch_size
    end_index = start_index + batch_size

    batches = poetrys_vector[start_index:end_index]
    length = max(map(len, batches))
    xdata = np.full((batch_size, length), word_num_map[' '], np.int32)
    for row in range(batch_size):
        xdata[row, :len(batches[row])] = batches[row]
    ydata = np.copy(xdata)
    ydata[:, :-1] = xdata[:, 1:]
    """
    xdata             ydata
    [6,2,4,6,9]       [2,4,6,9,9]
    [1,4,2,8,5]       [4,2,8,5,5]
    """
    x_batches.append(xdata)
    y_batches.append(ydata)

# ---------------------------------------RNN--------------------------------------#
#定义2个占位符，分别表示输入和预期输出
input_data = tf.placeholder(tf.int32, [batch_size, None])
output_targets = tf.placeholder(tf.int32, [batch_size, None])

keep_prob = 0.7
# 定义RNN
def neural_network(model='lstm', rnn_size=128, num_layers=2):
    if model == 'rnn':
        cell_fun = tf.nn.rnn_cell.BasicRNNCell
    elif model == 'gru':
        cell_fun = tf.nn.rnn_cell.GRUCell
    elif model == 'lstm':
        cell_fun = tf.nn.rnn_cell.BasicLSTMCell

    cell = cell_fun(rnn_size, state_is_tuple=True)
#state_is_tuple=True的时候，state是元组形式，state=(c,h)。
# 如果是False，那么state是一个由c和h拼接起来的张量，state=tf.concat(1,[c,h])
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
#2层的LSTM网络，前一层的LSTM的输出作为后一层的输入
    initial_state = cell.zero_state(batch_size, tf.float32)
#使用zero_state即可对各种状态进行初始化
# 如果是生成模式，output_data为None，则初始状态shape为[1 * rnn_size]
    with tf.variable_scope('rnnlm'):
        softmax_w = tf.get_variable("softmax_w", [rnn_size, len(words) + 1])
        softmax_b = tf.get_variable("softmax_b", [len(words) + 1])
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [len(words) + 1, rnn_size])# vocab size * hidden size
            inputs = tf.nn.embedding_lookup(embedding, input_data)

    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
    output = tf.reshape(outputs, [-1, rnn_size])

    logits = tf.matmul(output, softmax_w) + softmax_b
    probs = tf.nn.softmax(logits)
    return logits, last_state, probs, cell, initial_state


# -------------------------------生成古诗---------------------------------#
# 使用训练完成的模型
def to_word1(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    return words[sample]
def gen_poetry(begin_word=None):
    def to_word(weights):
        # t = np.cumsum(weights)
        # s = np.sum(weights)
        # sample = int(np.searchsorted(t, np.random.rand(1) * s))
        # return words[sample]

        sample = np.argmax(weights)
        if sample > len(words):
            print('error')
            sample = len(words) - 1
        return words[sample]

    _, last_state, probs, cell, initial_state = neural_network()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver(tf.all_variables())
        saver.restore(sess,'./trained_variables.ckpt-48')

        state_ = sess.run(cell.zero_state(1, tf.float32))

        x = np.array([list(map(word_num_map.get, '['))])
        [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
        # 如果指定开始的字
        begin_word_list=[s for s in begin_word]
        begin_word_list.reverse()
        poem = ''
        if begin_word_list:
            word = begin_word_list[-1]
        else:
            word = to_word1(probs_)
        while word != ']':
            if begin_word_list:
                word = begin_word_list.pop()
            # else:
            #     word = to_word(probs_)
            poem += word
            x = np.zeros((1, 1))
            x[0, 0] = word_num_map[word]
            [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
            word = to_word(probs_)
        # word = words[np.argmax(probs_)]
        return poem

# print(gen_poetry('我'))
# tf.reset_default_graph()
# print(gen_poetry('一'))
# tf.reset_default_graph()
print(gen_poetry(''))