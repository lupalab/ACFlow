import os
import pickle
import numpy as np
import tensorflow as tf

from .mask_generators import *

image_shape = [32, 32, 1]
data_path = '/data/omniglot/'
with open(data_path + 'train_vinyals_aug90.pkl', 'rb') as f:
    train_dict = pickle.load(f, encoding='bytes')
with open(data_path + 'val_vinyals_aug90.pkl', 'rb') as f:
    valid_dict = pickle.load(f, encoding='bytes')
with open(data_path + 'test_vinyals_aug90.pkl', 'rb') as f:
    test_dict = pickle.load(f, encoding='bytes')

mask_gen = OmniglotMaskGenerator()

def _parse_train(i):
    image = train_dict[b'images'][i]
    image = np.pad(image, ((2, 2), (2, 2), (0, 0)), mode='constant')
    mask = mask_gen(image)

    return image, mask


def _parse_valid(i):
    image = valid_dict[b'images'][i]
    image = np.pad(image, ((2, 2), (2, 2), (0, 0)), mode='constant')
    mask = mask_gen(image)

    return image, mask


def _parse_test(i):
    image = test_dict[b'images'][i]
    image = np.pad(image, ((2, 2), (2, 2), (0, 0)), mode='constant')
    mask = mask_gen(image)

    return image, mask


def get_dst(split):
    if split == 'train':
        size = len(train_dict[b'images'])
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.shuffle(size)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_train, [i],
                       [tf.uint8, tf.uint8])),
                      num_parallel_calls=16)
    elif split == 'valid':
        size = len(valid_dict[b'images'])
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_valid, [i],
                       [tf.uint8, tf.uint8])),
                      num_parallel_calls=16)
    else:
        size = len(test_dict[b'images'])
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_test, [i],
                       [tf.uint8, tf.uint8])),
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
        x, m = dst_it.get_next()
        self.x = tf.reshape(x, [batch_size] + image_shape)
        self.m = tf.reshape(m, [batch_size] + image_shape)
        self.image_shape = image_shape
        self.initializer = dst_it.initializer

    def initialize(self, sess):
        sess.run(self.initializer)

