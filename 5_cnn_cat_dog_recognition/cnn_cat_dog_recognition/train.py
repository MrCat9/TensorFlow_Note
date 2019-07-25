import dataset
import tensorflow as tf
# import time
# from datetime import timedelta
# import math
# import random
# import numpy as np
# conda install --channel https://conda.anaconda.org/menpo opencv3
# Adding Seed so that random initialization is consistent
from numpy.random import seed
from tensorflow import set_random_seed

seed(10)

set_random_seed(20)

batch_size = 32

# Prepare input data
classes = ['dogs', 'cats']
num_classes = len(classes)

# 20% of the data will automatically be used for validation
validation_size = 0.2
img_size = 64
num_channels = 3
train_path = 'training_data'

# We shall load all the training and validation images and labels into memory using openCV and use that during training
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)

print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))

session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')

# labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)  # 取每一行的最大值的索引

# Network graph params
filter_size_conv1 = 3
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64

fc_layer_size = 1024


def create_weights(shape):  # 初始化 W
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):  # 初始化 b
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(input,  # 卷积层
                               num_input_channels,
                               conv_filter_size,
                               num_filters):
    # We shall define the weights that will be trained using create_weights function. 3 3 3 32
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])  # 卷积核
    # We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)  # 偏置

    # Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,  # 卷积结果
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases  # 加上偏置

    layer = tf.nn.relu(layer)  # 通过激活函数

    # We shall be using max-pooling.
    layer = tf.nn.max_pool(value=layer,  # 池化
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    # Output of pooling is fed to Relu which is the activation function for us.
    # layer = tf.nn.relu(layer)

    return layer


def create_flatten_layer(layer):  # 与全连接层相乘之前需要先将输入“拉平”  # 平坦化
    # We know that the shape of the layer will be [batch_size img_size img_size num_channels]
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    # Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()  # 4096  # 4096=8*8*64

    # Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])  # (?, 4096)

    return layer


def create_fc_layer(input,  # 全连接层
                    num_inputs,
                    num_outputs,
                    use_relu=True):
    # Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])  # (4096, 1024)
    biases = create_biases(num_outputs)  # (1024,)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases  # (?, 1024)

    layer = tf.nn.dropout(layer, keep_prob=0.7)

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


layer_conv1 = create_convolutional_layer(input=x,  # 卷积层1  # 输出 (?, 32, 32, 32)
                                         num_input_channels=num_channels,
                                         conv_filter_size=filter_size_conv1,
                                         num_filters=num_filters_conv1)
layer_conv2 = create_convolutional_layer(input=layer_conv1,  # 输出 (?, 16, 16, 32)
                                         num_input_channels=num_filters_conv1,
                                         conv_filter_size=filter_size_conv2,
                                         num_filters=num_filters_conv2)

layer_conv3 = create_convolutional_layer(input=layer_conv2,  # 输出 (?, 8, 8, 64)
                                         num_input_channels=num_filters_conv2,
                                         conv_filter_size=filter_size_conv3,
                                         num_filters=num_filters_conv3)

layer_flat = create_flatten_layer(layer_conv3)  # 输出 (?, 4096)

layer_fc1 = create_fc_layer(input=layer_flat,  # 输出 (?, 1024)
                            num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                            num_outputs=fc_layer_size,
                            use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1,  # 输出 (?, 2)
                            num_inputs=fc_layer_size,
                            num_outputs=num_classes,
                            use_relu=False)

y_pred = tf.nn.softmax(layer_fc2, name='y_pred')  # (?, 2)

y_pred_cls = tf.argmax(y_pred, dimension=1)  # (?,)
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.global_variables_initializer())


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss, i):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0}--- iterations: {1}--- Training Accuracy: {2:>6.1%}, Validation Accuracy: {3:>6.1%},  Validation Loss: {4:.3f}"
    print(msg.format(epoch + 1, i, acc, val_acc, val_loss))


total_iterations = 0  # 用 1 个 batch 迭代一次是一次 iteration

saver = tf.train.Saver()  # 默认的 max_to_keep=5


def train(num_iteration):
    global total_iterations

    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)  # 训练集
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)  # 验证集

        feed_dict_tr = {x: x_batch,
                        y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                         y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples / batch_size) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples / batch_size))

            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss, i)
            saver.save(session, './dogs-cats-model/dog-cat.ckpt', global_step=i)

    total_iterations += num_iteration


train(num_iteration=8000)
