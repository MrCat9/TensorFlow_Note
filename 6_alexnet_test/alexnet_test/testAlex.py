#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import urllib.request
import argparse
import sys
import alexnet
import cv2
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv
import tensorflow as tf
import numpy as np
import caffe_classes

parser = argparse.ArgumentParser(description='Classify some images.')
parser.add_argument('mode', choices=['folder', 'url'], default='folder')
parser.add_argument('path', help='Specify a path [e.g. testModel]')
args = parser.parse_args(sys.argv[1:])  # 传入的参数

if args.mode == 'folder':
    # get testImage
    withPath = lambda f: '{}/{}'.format(args.path, f)  # 传入文件名f，拼接文件路径args.path与文件名f
    testImg = dict((f, cv2.imread(withPath(f))) for f in os.listdir(args.path) if os.path.isfile(withPath(f)))
elif args.mode == 'url':
    def url2img(url):
        '''url to image'''
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image


    testImg = {args.path: url2img(args.path)}

if testImg.values():
    # some params
    dropoutPro = 1  # 因为是直接使用网络，而不是训练，所以 dropout 的 keep_prob 比例为 1
    classNum = 1000  # 1000 分类
    skip = []  # 要跳过的层

    imgMean = np.array([104, 117, 124], np.float)  # R G B 三色的均值
    x = tf.placeholder("float", [1, 227, 227, 3])  # 输入  # 一次传入一张图片来测试

    model = alexnet.alexNet(x, dropoutPro, classNum, skip)
    score = model.fc3  # (1, 1000)
    softmax = tf.nn.softmax(score)  # (1, 1000)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.loadModel(sess)  # 往模型里传入权重值

        for key, img in testImg.items():
            # img preprocess
            resized = cv2.resize(img.astype(np.float), (227, 227)) - imgMean  # 因为该模型在训练的时候有做减均值的操作，所以在测试的时候也需要减均值  # shape=(227, 227, 3)
            maxx = np.argmax(sess.run(softmax, feed_dict={x: resized.reshape((1, 227, 227, 3))}))  # 得到 softmax 结果中最大值的索引
            res = caffe_classes.class_names[maxx]  # 分类结果

            # 将结果写到图片上
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, res, (int(img.shape[0] / 3), int(img.shape[1] / 3)), font, 1, (0, 255, 0), 2)
            print("{}: {}\n----".format(key, res))
            cv2.imshow("demo", img)
            cv2.waitKey(0)
