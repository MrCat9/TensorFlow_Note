# -*- coding: utf-8 -*-


import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np


def load_test(test_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []

    print('Going to read testing images')
    for fields in classes:
        index = classes.index(fields)
        print('Now going to read {} files (Index: {})'.format(fields, index))
        path = os.path.join(test_path, fields, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            # import matplotlib.pyplot as plt
            # plt.imshow(image)
            # plt.show()
            image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)  # shape=(64, 64, 3)
            # plt.imshow(image)
            # plt.show()
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0  # shape=(2,)
            labels.append(label)
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            cls.append(fields)
    images = np.array(images)  # shape=(1000, 64, 64, 3)
    labels = np.array(labels)  # shape=(1000, 2)
    img_names = np.array(img_names)  # shape=(1000,)
    cls = np.array(cls)  # shape=(1000,)

    return images, labels, img_names, cls


class DataSet(object):

    def __init__(self, images, labels, img_names, cls):
        self._num_examples = images.shape[0]

        self._images = images
        self._labels = labels
        self._img_names = img_names
        self._cls = cls
        self._epochs_done = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def img_names(self):
        return self._img_names

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # After each epoch we update this
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


def read_test_sets(test_path, image_size, classes):
    class DataSets(object):
        pass

    data_sets = DataSets()

    images, labels, img_names, cls = load_test(test_path, image_size, classes)
    images, labels, img_names, cls = shuffle(images, labels, img_names, cls)

    # 测试集
    test_images = images
    test_labels = labels
    test_img_names = img_names
    test_cls = cls

    data_sets.test = DataSet(test_images, test_labels, test_img_names, test_cls)

    return data_sets
