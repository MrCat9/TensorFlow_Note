# coding: UTF-8


import tensorflow as tf
import numpy as np


# define different layer functions
# we usually don't do convolution and pooling on batch and channel
def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding="SAME"):
    """max-pooling"""
    return tf.nn.max_pool(x, ksize=[1, kHeight, kWidth, 1],
                          strides=[1, strideX, strideY, 1], padding=padding, name=name)


def dropout(x, keepPro, name=None):
    """dropout"""
    return tf.nn.dropout(x, keepPro, name)


def LRN(x, R, alpha, beta, name=None, bias=1.0):
    """LRN"""
    # Local Response Normalization 局部响应归一化
    return tf.nn.local_response_normalization(x, depth_radius=R, alpha=alpha,
                                              beta=beta, bias=bias, name=name)


def fcLayer(x, inputD, outputD, reluFlag, name):
    """fully-connect"""
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape=[inputD, outputD], dtype="float")
        b = tf.get_variable("b", [outputD], dtype="float")
        out = tf.nn.xw_plus_b(x, w, b, name=scope.name)  # x * w + b
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out


def convLayer(x, kHeight, kWidth, strideX, strideY,
              featureNum, name, padding="SAME", groups=1):
    """convlutional"""
    channel = int(x.get_shape()[-1])
    conv = lambda a, b: tf.nn.conv2d(a, b, strides=[1, strideY, strideX, 1], padding=padding)  # a 和 b 做卷积
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape=[kHeight, kWidth, channel / groups, featureNum])  # groups 表示切分为几组
        b = tf.get_variable("b", shape=[featureNum])

        xNew = tf.split(value=x, num_or_size_splits=groups, axis=3)  # 沿着第4个纬度进行切分，切成 groups 份
        wNew = tf.split(value=w, num_or_size_splits=groups, axis=3)

        featureMap = [conv(t1, t2) for t1, t2 in zip(xNew, wNew)]  # 切分的几个部分分别做卷积
        mergeFeatureMap = tf.concat(axis=3, values=featureMap)  # 合并卷积结果
        # print mergeFeatureMap.shape
        out = tf.nn.bias_add(mergeFeatureMap, b)
        return tf.nn.relu(tf.reshape(out, mergeFeatureMap.get_shape().as_list()), name=scope.name)


class alexNet(object):
    """alexNet model"""

    def __init__(self, x, keepPro, classNum, skip, modelPath="bvlc_alexnet.npy"):
        self.X = x  # shape=(1, 227, 227, 3)
        self.KEEPPRO = keepPro
        self.CLASSNUM = classNum
        self.SKIP = skip
        self.MODELPATH = modelPath
        # build CNN
        self.buildCNN()

    def buildCNN(self):
        """build model"""
        conv1 = convLayer(self.X, 11, 11, 4, 4, 96, "conv1", "VALID")  # shape=(1, 55, 55, 96)
        pool1 = maxPoolLayer(conv1, 3, 3, 2, 2, "pool1", "VALID")  # (1, 27, 27, 96)
        lrn1 = LRN(pool1, 2, 2e-05, 0.75, "norm1")  # (1, 27, 27, 96)

        conv2 = convLayer(lrn1, 5, 5, 1, 1, 256, "conv2", groups=2)  # (1, 27, 27, 256)
        pool2 = maxPoolLayer(conv2, 3, 3, 2, 2, "pool2", "VALID")  # (1, 13, 13, 256)
        lrn2 = LRN(pool2, 2, 2e-05, 0.75, "lrn2")  # (1, 13, 13, 256)

        conv3 = convLayer(lrn2, 3, 3, 1, 1, 384, "conv3")  # (1, 13, 13, 384)

        conv4 = convLayer(conv3, 3, 3, 1, 1, 384, "conv4", groups=2)  # (1, 13, 13, 384)

        conv5 = convLayer(conv4, 3, 3, 1, 1, 256, "conv5", groups=2)  # (1, 13, 13, 256)
        pool5 = maxPoolLayer(conv5, 3, 3, 2, 2, "pool5", "VALID")  # (1, 6, 6, 256)

        fcIn = tf.reshape(pool5, [-1, 256 * 6 * 6])  # 平坦化  # (1, 9216)
        fc1 = fcLayer(fcIn, 256 * 6 * 6, 4096, True, "fc6")  # (1, 4096)
        dropout1 = dropout(fc1, self.KEEPPRO)

        fc2 = fcLayer(dropout1, 4096, 4096, True, "fc7")  # (1, 4096)
        dropout2 = dropout(fc2, self.KEEPPRO)

        self.fc3 = fcLayer(dropout2, 4096, self.CLASSNUM, True, "fc8")  # (1, 1000)

    def loadModel(self, sess):  # 往模型里传入权重值
        """load model"""
        wDict = np.load(self.MODELPATH, encoding="bytes").item()
        # for layers in model
        for name in wDict:  # 迭代出每一层的权重参数
            if name not in self.SKIP:  # 不是需要跳过的层
                with tf.variable_scope(name, reuse=True):
                    for p in wDict[name]:
                        if len(p.shape) == 1:  # 等于1的是偏置的权重值
                            # bias
                            sess.run(tf.get_variable('b', trainable=False).assign(p))
                        else:
                            # weights
                            sess.run(tf.get_variable('w', trainable=False).assign(p))
