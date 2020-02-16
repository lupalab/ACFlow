import os
import tensorflow as tf
import numpy as np
from PIL import Image

from .mask_generators import *

image_shape = [64, 64, 3]
path = '/data/celebA/img/'
list_file = '/data/celebA/Eval/list_eval_partition.txt'
with open(list_file, 'r') as f:
    all_list = f.readlines()
all_list = [e.strip().split() for e in all_list]
train_list = [e[0] for e in all_list if e[1] == '0']
train_list = [path + e for e in train_list]
valid_list = [e[0] for e in all_list if e[1] == '1']
valid_list = [path + e for e in valid_list]
test_list = [e[0] for e in all_list if e[1] == '2']
test_list = [path + e for e in test_list]

mask_gen = CelebAMaskGenerator()


def _parse_func(f):
    image = Image.open(f).crop((25, 45, 153, 173)).resize((64, 64))
    image = np.array(image).astype('uint8')
    mask = mask_gen(image)

    return image, mask


def get_dst(split):
    if split == 'train':
        size = len(train_list)
        files = tf.constant(train_list, dtype=tf.string)
        dst = tf.data.Dataset.from_tensor_slices(files)
        dst = dst.shuffle(size)
    elif split == 'valid':
        size = len(valid_list)
        files = tf.constant(valid_list, dtype=tf.string)
        dst = tf.data.Dataset.from_tensor_slices(files)
    else:
        size = len(test_list)
        files = tf.constant(test_list, dtype=tf.string)
        dst = tf.data.Dataset.from_tensor_slices(files)

    dst = dst.map(lambda f: tuple(
        tf.py_func(_parse_func, [f], [tf.uint8, tf.uint8])),
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

