import numpy as np
import tensorflow as tf


def preprocess(x, m, data_constraint=0.9, n_bits=8, rand=True):
    '''
    Args:
        x: [B,H,W,C] [uint8]
        m: [B,H,W,C] [float32]
    '''
    x = tf.cast(x, tf.float32)
    if n_bits < 8:
        x = tf.floor(x / 2 ** (8 - n_bits))
    n_bins = 2**n_bits
    if rand:
        x = (x + tf.random_uniform(tf.shape(x))) / n_bins
    else:
        x = x / n_bins
    pre_logit_scale = np.log(data_constraint)
    pre_logit_scale -= np.log(1. - data_constraint)
    pre_logit_scale = tf.cast(pre_logit_scale, tf.float32)
    logit_x = 2. * x  # [0, 2]
    logit_x -= 1.  # [-1, 1]
    logit_x *= data_constraint  # [-.9, .9]
    logit_x += 1.  # [.1, 1.9]
    logit_x /= 2.  # [.05, .95]
    logit_x = tf.log(logit_x) - tf.log(1. - logit_x)
    objective = tf.nn.softplus(logit_x) + \
        tf.nn.softplus(-logit_x) - \
        tf.nn.softplus(-pre_logit_scale) - \
        np.log(n_bins)
    objective = tf.reduce_sum(objective * (1.-m), [1, 2, 3])

    return logit_x, objective


def postprocess(logit_x, m, data_constraint=0.9, n_bits=8):
    n_bins = 2**n_bits
    x = tf.nn.sigmoid(logit_x)
    x *= 2.
    x -= 1.
    x /= data_constraint
    x += 1.
    x /= 2.
    x *= n_bins
    if n_bits < 8:
        x = x * 2 ** (8 - n_bits)
    x = tf.clip_by_value(x, 0., 255.)

    pre_logit_scale = np.log(data_constraint)
    pre_logit_scale -= np.log(1. - data_constraint)
    pre_logit_scale = tf.cast(pre_logit_scale, tf.float32)
    objective = tf.nn.softplus(logit_x) + \
        tf.nn.softplus(-logit_x) - \
        tf.nn.softplus(-pre_logit_scale) - \
        np.log(n_bins)
    objective = tf.reduce_sum(objective * (1.-m), [1, 2, 3])
    objective *= -1

    return x, objective
