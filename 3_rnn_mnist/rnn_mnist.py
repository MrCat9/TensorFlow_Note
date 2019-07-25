# -*- coding: utf-8 -*-
# tensorflow 实现 rnn 识别手写数字


import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


sess = tf.Session()
mnist = input_data.read_data_sets('data', one_hot=True)
print(mnist.train.images.shape)

lr = 1e-3  # 学习率
input_size = 28  # 每行输入28个特征点
timestep_size = 28  # 持续输入28行
hidden_size = 256  # 隐含层的数量
layer_num = 2  # LSTM layer 的层数
class_num = 10  # 10分类问题

_X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, class_num])

batch_size = tf.placeholder(tf.int32, [])
keep_prob = tf.placeholder(tf.float32, [])

X = tf.reshape(_X, [-1, 28, 28])  # batch, timestep_size, input_size


def lstm_cell():  # 一层 LSTM
    cell = rnn.LSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
    return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)


mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple=True)

# 用全零来初始化状态
init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

# 得到每一层的输出结果
outputs = list()
state = init_state
with tf.variable_scope('RNN'):
    for timestep in range(timestep_size):
        if timestep > 0:
            tf.get_variable_scope().reuse_variables()
        (cell_output, state) = mlstm_cell(X[:, timestep, :], state)  # 输入一行的28个特征点
        outputs.append(cell_output)
h_state = outputs[-1]  # shape=(?, 256)

# Softmax 层参数
W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)  # 全连接层  # shape=(256, 10)
bias = tf.Variable(tf.constant(0.1, shape=[class_num]), dtype=tf.float32)
y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)

# 损失和评估函数
cross_entropy = -tf.reduce_mean(y * tf.log(y_pre))
train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y, 1))  # shape=(?,)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.global_variables_initializer())
for i in range(1000):
    _batch_size = 128
    batch = mnist.train.next_batch(_batch_size)
    if (i + 1) % 200 == 0:  # 200 次迭代 打印一次
        train_accuracy = sess.run(accuracy, feed_dict={
            _X: batch[0], y: batch[1], keep_prob: 1.0, batch_size: _batch_size})
        # 已经迭代完成的 epoch 数: mnist.train.epochs_completed
        print("Iter%d, step %d, training accuracy %g" % (mnist.train.epochs_completed, (i + 1), train_accuracy))
    sess.run(train_op, feed_dict={_X: batch[0], y: batch[1], keep_prob: 0.5, batch_size: _batch_size})

# 计算测试数据的准确率
print("test accuracy %g" % sess.run(accuracy, feed_dict={
    _X: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0, batch_size: mnist.test.images.shape[0]}))

#


# ================ 单个图像RNN每层结果 ================
_batch_size = 5
X_batch, y_batch = mnist.test.next_batch(_batch_size)
print(X_batch.shape, y_batch.shape)
_outputs, _state = sess.run([outputs, state], feed_dict={
    _X: X_batch, y: y_batch, keep_prob: 1.0, batch_size: _batch_size})
print('_outputs.shape =', np.asarray(_outputs).shape)  # _outputs.shape = (28, 5, 256)  (timestep_每张图片切成28条，batch，神经元个数)


# img_no = 4
print(mnist.train.labels[4])

X3 = mnist.train.images[4]  # shape=(784,)
img3 = X3.reshape([28, 28])  # shape=(28, 28)
plt.imshow(img3, cmap='gray')
plt.show()

X3.shape = [-1, 784]  # shape=(1, 784)
y_batch = mnist.train.labels[0]  # shape=(10,)  # 只是用于占位
y_batch.shape = [-1, class_num]  # shape=(1, 10)

X3_outputs = np.array(sess.run(outputs, feed_dict={
    _X: X3, y: y_batch, keep_prob: 1.0, batch_size: 1}))
print(X3_outputs.shape)  # (28, 1, 256)
X3_outputs.shape = [28, hidden_size]  # shape=(28, 256)
print(X3_outputs.shape)  # (28, 256)

h_W = sess.run(W, feed_dict={  # 全连接层
    _X: X3, y: y_batch, keep_prob: 1.0, batch_size: 1})  # shape=(256, 10)
h_bias = sess.run(bias, feed_dict={
    _X: X3, y: y_batch, keep_prob: 1.0, batch_size: 1})  # shape=(10,)
h_bias.shape = [-1, 10]  # shape=(1, 10)

bar_index = range(class_num)  # 条形图的 x 轴
for i in range(X3_outputs.shape[0]):
    plt.subplot(7, 4, i + 1)
    X3_h_shate = X3_outputs[i, :].reshape([-1, hidden_size])  # shape=(1, 256)  # 中间结果
    pro = sess.run(tf.nn.softmax(tf.matmul(X3_h_shate, h_W) + h_bias))  # shape=(1, 10)  # 中间结果的分类结果
    plt.bar(bar_index, pro[0], width=0.2, align='center')
    plt.axis('off')
plt.show()
