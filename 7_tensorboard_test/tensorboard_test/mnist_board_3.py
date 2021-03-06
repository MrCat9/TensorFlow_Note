# -*- coding: utf-8 -*-
# 使用 name_scope 和 name 进行命名
# 用 tf.summary.histogram() 看分布  # 用 tf.summary.image() 拿几个数据出来看  # tf.summary.scalar() 画图  # tf.summary.merge_all() 将要显示的参数聚到一起


# import os
import tensorflow as tf
# import urllib

LOGDIR = './mnist/'

mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=LOGDIR + 'data', one_hot=True)


def conv_layer(input, size_in, size_out, name="conv"):
    with tf.name_scope(name):
        # w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")  # 随机初始化
        w = tf.Variable(tf.zeros([5, 5, size_in, size_out]), name="W")  # 全零初始化
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def fc_layer(input, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        act = tf.nn.relu(tf.matmul(input, w) + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act


def mnist_model(learning_rate, use_two_conv, use_two_fc, hparam):
    tf.reset_default_graph()
    sess = tf.Session()

    # Setup placeholders, and reshape the data
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")  # (?, 784)
    x_image = tf.reshape(x, [-1, 28, 28, 1])  # (?, 28, 28, 1)
    tf.summary.image('input', x_image, 3)
    y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")  # (?, 10)

    if use_two_conv:
        conv1 = conv_layer(x_image, 1, 32, "conv1")  # (?, 14, 14, 32)
        conv_out = conv_layer(conv1, 32, 64, "conv2")  # (?, 7, 7, 64)
    else:
        conv1 = conv_layer(x_image, 1, 64, "conv")
        conv_out = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64])  # (?, 3136)

    if use_two_fc:
        fc1 = fc_layer(flattened, 7 * 7 * 64, 1024, "fc1")  # (?, 1024)
        embedding_input = fc1
        embedding_size = 1024
        logits = fc_layer(fc1, 1024, 10, "fc2")  # (?, 10)
    else:
        embedding_input = flattened
        embedding_size = 7 * 7 * 64
        logits = fc_layer(flattened, 7 * 7 * 64, 10, "fc")

    with tf.name_scope("loss"):
        xent = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=y), name="loss")
        tf.summary.scalar("loss", xent)

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    summ = tf.summary.merge_all()

    embedding = tf.Variable(tf.zeros([1024, embedding_size]), name="test_embedding")
    assignment = embedding.assign(embedding_input)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    tenboard_dir = './tensorboard/test3/'
    writer = tf.summary.FileWriter(tenboard_dir + hparam)
    writer.add_graph(sess.graph)

    for i in range(2001):
        batch = mnist.train.next_batch(100)
        if i % 5 == 0:
            [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batch[0], y: batch[1]})
            writer.add_summary(s, i)
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})


def make_hparam_string(learning_rate, use_two_fc, use_two_conv):
    conv_param = "conv=2" if use_two_conv else "conv=1"
    fc_param = "fc=2" if use_two_fc else "fc=1"
    return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)


def main():
    # You can try adding some more learning rates
    for learning_rate in [1E-4]:

        # Include "False" as a value to try different model architectures
        for use_two_fc in [True]:
            for use_two_conv in [True]:
                # Construct a hyperparameter string for each one (example: "lr_1E-3,fc=2,conv=2)
                hparam = make_hparam_string(learning_rate, use_two_fc, use_two_conv)
                print('Starting run for %s' % hparam)

                # Actually run with the new settings
                mnist_model(learning_rate, use_two_fc, use_two_conv, hparam)


if __name__ == '__main__':
    main()
