# -*- coding: utf-8 -*-
# tensorflow 实现 cnn 完成图像10分类


import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf


def model():
    x_reshaped = tf.reshape(x, shape=[-1, 24, 24, 1])  # batch*24*24*1

    conv_out1 = conv_layer(x_reshaped, W1, b1)  # batch*24*24*64
    maxpool_out1 = maxpool_layer(conv_out1)  # batch*12*12*64
    # 提出了LRN层，对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力。
    # 推荐阅读http://blog.csdn.net/banana1006034246/article/details/75204013
    norm1 = tf.nn.lrn(maxpool_out1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)  # batch*12*12*64
    conv_out2 = conv_layer(norm1, W2, b2)  # batch*12*12*64
    norm2 = tf.nn.lrn(conv_out2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)  # batch*12*12*64
    maxpool_out2 = maxpool_layer(norm2)  # batch*6*6*64

    maxpool_reshaped = tf.reshape(maxpool_out2, [-1, W3.get_shape().as_list()[0]])  # batch*2304  # (batch, 6*6*64)
    local = tf.add(tf.matmul(maxpool_reshaped, W3), b3)  # batch*1024
    local_out = tf.nn.relu(local)

    out = tf.add(tf.matmul(local_out, W_out), b_out)  # batch*10
    return out


def conv_layer(x, W, b):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    conv_with_b = tf.nn.bias_add(conv, b)
    conv_out = tf.nn.relu(conv_with_b)
    return conv_out


def maxpool_layer(conv, k=2):
    return tf.nn.max_pool(conv, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def show_conv_results(data, filename=None):
    # 查看卷积后得到的特征图
    plt.figure()
    rows, cols = 4, 8  # 1 张图片有 32 个 24*24 的特征图
    for i in range(np.shape(data)[3]):
        img = data[0, :, :, i]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='Greys_r', interpolation='none')
        plt.axis('off')
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def show_weights(W, filename=None):
    # 查看权重矩阵
    plt.figure()
    rows, cols = 4, 8  # 32 个 5*5*1 的权重矩阵
    for i in range(np.shape(W)[3]):
        img = W[:, :, 0, i]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='Greys_r', interpolation='none')
        plt.axis('off')
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def show_some_examples(names, data, labels):
    plt.figure()
    rows, cols = 4, 4  # 共 16 个子图
    random_idxs = random.sample(range(len(data)), rows * cols)  # 随机生成 rows * cols 个索引
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        j = random_idxs[i]
        plt.title(names[labels[j]])  # 子图的标题为该图片的标签
        img = np.reshape(data[j, :], (24, 24))
        plt.imshow(img, cmap='Greys_r')
        plt.axis('off')
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域。
    # plt.show()
    plt.savefig('cifar_examples.png')


def clean(data):
    # 对图片进行预处理
    imgs = data.reshape(data.shape[0], 3, 32, 32)  # 50000*3*32*32
    grayscale_imgs = imgs.mean(1)  # 取 RGB 三通道的平均值，将彩色图变成灰度图  # 50000*32*32
    cropped_imgs = grayscale_imgs[:, 4:28, 4:28]  # 裁剪  # 50000*24*24
    img_data = cropped_imgs.reshape(data.shape[0], -1)  # 50000*576
    img_size = np.shape(img_data)[1]  # 一张图片的大小
    means = np.mean(img_data, axis=1)  # 对每一行取平均  # 1*50000  # 压缩列
    meansT = means.reshape(len(means), 1)  # 50000*1
    stds = np.std(img_data, axis=1)  # 对每一行求标准差  # 1*50000
    stdsT = stds.reshape(len(stds), 1)  # 50000*1
    adj_stds = np.maximum(stdsT, 1.0 / np.sqrt(img_size))  # np.sqrt() 求平方根
    normalized = (img_data - meansT) / adj_stds  # 标准化
    return normalized


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict


def read_data(directory):
    names = unpickle('{}/batches.meta'.format(directory))['label_names']
    print('names', names)

    data, labels = [], []
    for i in range(1, 6):
        filename = '{}/data_batch_{}'.format(directory, i)
        batch_data = unpickle(filename)  # 一个 batch 里面有 10000 张图片(32*32*3)
        if len(data) > 0:
            data = np.vstack((data, batch_data['data']))
            labels = np.hstack((labels, batch_data['labels']))
        else:
            data = batch_data['data']
            labels = batch_data['labels']

    print(np.shape(data), np.shape(labels))

    data = clean(data)
    data = data.astype(np.float32)
    return names, data, labels


if __name__ == '__main__':
    # random.seed(1)
    #
    # names, data, labels = read_data('./cifar-10-batches-py')  # 读数据
    # show_some_examples(names, data, labels)  # 显示几张图片
    #
    # raw_data = data[4, :]  # 取出第5张图片
    # raw_img = np.reshape(raw_data, (24, 24))
    # plt.figure()
    # plt.imshow(raw_img, cmap='Greys_r')
    # plt.show()
    #
    # x = tf.reshape(raw_data, shape=[-1, 24, 24, 1])  # 试着输入1张图片  # 1*24*24*1
    # W = tf.Variable(tf.random_normal([5, 5, 1, 32]))
    # b = tf.Variable(tf.random_normal([32]))
    #
    # conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # 卷积
    # conv_with_b = tf.nn.bias_add(conv, b)  # 加上偏置
    # conv_out = tf.nn.relu(conv_with_b)  # 通过 relu 函数
    #
    # k = 2
    # maxpool = tf.nn.max_pool(conv_out, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')  # 池化
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #
    #     W_val = sess.run(W)
    #     print('weights:')
    #     show_weights(W_val)
    #
    #     conv_val = sess.run(conv)
    #     print('convolution results:')
    #     print(np.shape(conv_val))
    #     show_conv_results(conv_val)
    #
    #     conv_out_val = sess.run(conv_out)
    #     print('convolution with bias and relu:')
    #     print(np.shape(conv_out_val))
    #     show_conv_results(conv_out_val)
    #
    #     maxpool_val = sess.run(maxpool)
    #     print('maxpool after all the convolutions:')
    #     print(np.shape(maxpool_val))
    #     show_conv_results(maxpool_val)

    # ================================================================

    names, data, labels = read_data('./cifar-10-batches-py')  # 读数据
    # names  list  len=10
    # data  shape=(50000, 576)
    # labels  shape=(50000,)

    x = tf.placeholder(tf.float32, [None, 24 * 24])
    y = tf.placeholder(tf.float32, [None, len(names)])
    W1 = tf.Variable(tf.random_normal([5, 5, 1, 64]))
    b1 = tf.Variable(tf.random_normal([64]))
    W2 = tf.Variable(tf.random_normal([5, 5, 64, 64]))
    b2 = tf.Variable(tf.random_normal([64]))
    W3 = tf.Variable(tf.random_normal([6 * 6 * 64, 1024]))  # 全连接层
    b3 = tf.Variable(tf.random_normal([1024]))
    W_out = tf.Variable(tf.random_normal([1024, len(names)]))  # 全连接层
    b_out = tf.Variable(tf.random_normal([len(names)]))

    learning_rate = 0.001
    model_op = model()

    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=model_op, labels=y)
    )
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(model_op, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        onehot_labels = tf.one_hot(labels, len(names), axis=-1)
        onehot_vals = sess.run(onehot_labels)  # shape=(50000, 10)
        batch_size = 64
        print('batch size', batch_size)
        for j in range(0, 1000):  # 1000 次 Epoch
            avg_accuracy_val = 0.  # 本次 Epoch 的平均准确率
            batch_count = 0.  # 本次 Epoch 已经使用了几个 batch
            for i in range(0, len(data), batch_size):  # range(起, 止, 步长)
                batch_data = data[i:i + batch_size, :]  # shape=(64, 576)
                batch_onehot_vals = onehot_vals[i:i + batch_size, :]  # shape=(64, 10)
                _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: batch_data, y: batch_onehot_vals})
                avg_accuracy_val += accuracy_val  # 本次 Epoch 的总准确率
                batch_count += 1.  # 本次 Epoch 已经使用了几个 batch
                if batch_count % 150 == 0:
                    print('    Epoch {}. batch {} accuracy {}'.format(j, batch_count, accuracy_val))
            avg_accuracy_val /= batch_count  # 本次 Epoch 的平均准确率
            print('Epoch {}. Avg accuracy {}'.format(j, avg_accuracy_val))

        # 使用测试集
        batch_data = unpickle('cifar-10-batches-py/test_batch')  # 一个 batch 里面有 10000 张图片(32*32*3)
        data = batch_data['data']  # <class 'tuple'>: (10000, 3072)
        labels = np.array(batch_data['labels'])  # <class 'tuple'>: (10000,)
        data = clean(data)  # <class 'tuple'>: (10000, 576)
        data = data.astype(np.float32)

        onehot_labels = tf.one_hot(labels, len(names), axis=-1)
        onehot_vals = sess.run(onehot_labels)  # shape=(10000, 10)
        batch_size = 64
        print('batch size', batch_size)
        for j in range(0, 1):  # 1 次 Epoch
            avg_accuracy_val = 0.  # 本次 Epoch 的平均准确率
            batch_count = 0.  # 本次 Epoch 已经使用了几个 batch
            for i in range(0, len(data), batch_size):  # range(起, 止, 步长)
                batch_data = data[i:i + batch_size, :]  # shape=(64, 576)
                batch_onehot_vals = onehot_vals[i:i + batch_size, :]  # shape=(64, 10)
                accuracy_val = sess.run(accuracy, feed_dict={x: batch_data, y: batch_onehot_vals})
                avg_accuracy_val += accuracy_val  # 本次 Epoch 的总准确率
                batch_count += 1.  # 本次 Epoch 已经使用了几个 batch
                print('    Epoch {}. batch {} accuracy {}'.format(j, batch_count, accuracy_val))
            avg_accuracy_val /= batch_count  # 本次 Epoch 的平均准确率
            print('Epoch {}. Avg accuracy {}'.format(j, avg_accuracy_val))
