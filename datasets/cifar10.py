import os
import numpy as np
import pickle
import tensorflow as tf

from .mask_generators import *

image_shape = [32, 32, 3]
data_path = '/data/cifar-10-batches-py/'
train_list = ['data_batch_1', 'data_batch_2',
              'data_batch_3', 'data_batch_4', 'data_batch_5']
test_list = ['test_batch']

data = []
target = []
for file_name in train_list:
    file_path = data_path + file_name
    with open(file_path, 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
    data.append(entry['data'])
    if 'labels' in entry:
        target.extend(entry['labels'])
    else:
        target.extend(entry['fine_labels'])
data = np.vstack(data).reshape(-1, 3, 32, 32)
train_x = data.transpose(0, 2, 3, 1).astype(np.uint8)
train_y = np.asarray(target).astype(np.int64)

data = []
target = []
for file_name in test_list:
    file_path = data_path + file_name
    with open(file_path, 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
    data.append(entry['data'])
    if 'labels' in entry:
        target.extend(entry['labels'])
    else:
        target.extend(entry['fine_labels'])
data = np.vstack(data).reshape(-1, 3, 32, 32)
test_x = data.transpose(0, 2, 3, 1).astype(np.uint8)
test_y = np.asarray(target).astype(np.int64)


def sample_valid(x, y):
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    x, y = x[idx], y[idx]

    x1, y1 = x[:-5000], y[:-5000]
    x2, y2 = x[-5000:], y[-5000:]

    return x1, y1, x2, y2

train_x, train_y, valid_x, valid_y = sample_valid(train_x, train_y)

mask_gen = CifarMaskGenerator()


def _parse_train(i):
    image = train_x[i]
    label = train_y[i]
    mask = mask_gen(image)

    return image, label, mask

def _parse_valid(i):
    image = valid_x[i]
    label = valid_y[i]
    mask = mask_gen(image)

    return image, label, mask


def _parse_test(i):
    image = test_x[i]
    label = test_y[i]
    mask = mask_gen(image)

    return image, label, mask


def get_dst(split):
    if split == 'train':
        size = train_x.shape[0]
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.shuffle(size)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_train, [i],
                       [tf.uint8, tf.int64, tf.uint8])),
                      num_parallel_calls=16)
    elif split == 'valid':
        size = valid_x.shape[0]
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_valid, [i],
                       [tf.uint8, tf.int64, tf.uint8])),
                      num_parallel_calls=16)
    else:
        size = test_x.shape[0]
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_test, [i],
                       [tf.uint8, tf.int64, tf.uint8])),
                      num_parallel_calls=16)

    return dst, size


class Dataset(object):
    def __init__(self, split, batch_size):
        dst, size = get_dst(split)
        self.size = size
        self.num_steps = self.size // batch_size
        dst = dst.batch(batch_size, drop_remainder=True)
        dst = dst.prefetch(1)

        dst_it = dst.make_initializable_iterator()
        x, y, m = dst_it.get_next()
        self.x = tf.reshape(x, [batch_size] + image_shape)
        self.y = tf.reshape(y, [batch_size])
        self.m = tf.reshape(m, [batch_size] + image_shape)
        self.image_shape = image_shape
        self.initializer = dst_it.initializer

    def initialize(self, sess):
        sess.run(self.initializer)
