import os
import numpy as np
import torch
import tensorflow as tf

from .mask_generators import *

image_shape = [32, 32, 1]
data_path = '/data/mnist/'
train_x, train_y = torch.load(data_path + 'training.pt')
train_x, train_y = train_x.numpy(), train_y.numpy()
train_x = np.pad(train_x, ((0, 0), (2, 2), (2, 2)), mode='constant')
train_x = train_x[:, :, :, np.newaxis]
test_x, test_y = torch.load(data_path + 'test.pt')
test_x, test_y = test_x.numpy(), test_y.numpy()
test_x = np.pad(test_x, ((0, 0), (2, 2), (2, 2)), mode='constant')
test_x = test_x[:, :, :, np.newaxis]

def sample_valid(x, y):
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    x, y = x[idx], y[idx]

    x1, y1 = x[:-5000], y[:-5000]
    x2, y2 = x[-5000:], y[-5000:]

    return x1, y1, x2, y2

train_x, train_y, valid_x, valid_y = sample_valid(train_x, train_y)

mask_gen = MnistMaskGenerator()


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
