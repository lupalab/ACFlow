from __future__ import print_function

import time
from datetime import datetime
import os

import numpy
from six.moves import xrange
import tensorflow as tf

from tensorflow import gfile

from .utils import *

_recursion_type = 2


# RESNET UTILS
def residual_block(input_, dim, name, use_batch_norm=True,
                   train=True, weight_norm=True, bottleneck=False):
    """Residual convolutional block."""
    with tf.variable_scope(name):
        res = input_
        if use_batch_norm:
            res = batch_norm(
                input_=res, dim=dim, name="bn_in", scale=False,
                train=train, epsilon=1e-4, axes=[0, 1, 2])
        res = tf.nn.relu(res)
        if bottleneck:
            res = conv_layer(
                input_=res, filter_size=[1, 1], dim_in=dim, dim_out=dim,
                name="h_0", stddev=numpy.sqrt(2. / (dim)),
                strides=[1, 1, 1, 1], padding="SAME",
                nonlinearity=None, bias=(not use_batch_norm),
                weight_norm=weight_norm, scale=False)
            if use_batch_norm:
                res = batch_norm(
                    input_=res, dim=dim,
                    name="bn_0", scale=False, train=train,
                    epsilon=1e-4, axes=[0, 1, 2])
            res = tf.nn.relu(res)
            res = conv_layer(
                input_=res, filter_size=[3, 3], dim_in=dim,
                dim_out=dim, name="h_1", stddev=numpy.sqrt(2. / (1. * dim)),
                strides=[1, 1, 1, 1], padding="SAME", nonlinearity=None,
                bias=(not use_batch_norm),
                weight_norm=weight_norm, scale=False)
            if use_batch_norm:
                res = batch_norm(
                    input_=res, dim=dim, name="bn_1", scale=False,
                    train=train, epsilon=1e-4, axes=[0, 1, 2])
            res = tf.nn.relu(res)
            res = conv_layer(
                input_=res, filter_size=[1, 1], dim_in=dim, dim_out=dim,
                name="out", stddev=numpy.sqrt(2. / (1. * dim)),
                strides=[1, 1, 1, 1], padding="SAME", nonlinearity=None,
                bias=True, weight_norm=weight_norm, scale=True)
        else:
            res = conv_layer(
                input_=res, filter_size=[3, 3], dim_in=dim, dim_out=dim,
                name="h_0", stddev=numpy.sqrt(2. / (dim)),
                strides=[1, 1, 1, 1], padding="SAME",
                nonlinearity=None, bias=(not use_batch_norm),
                weight_norm=weight_norm, scale=False)
            if use_batch_norm:
                res = batch_norm(
                    input_=res, dim=dim, name="bn_0", scale=False,
                    train=train, epsilon=1e-4, axes=[0, 1, 2])
            res = tf.nn.relu(res)
            res = conv_layer(
                input_=res, filter_size=[3, 3], dim_in=dim, dim_out=dim,
                name="out", stddev=numpy.sqrt(2. / (1. * dim)),
                strides=[1, 1, 1, 1], padding="SAME", nonlinearity=None,
                bias=True, weight_norm=weight_norm, scale=True)
        res += input_

    return res


def resnet(input_, dim_in, dim, dim_out, name, use_batch_norm=True,
           train=True, weight_norm=True, residual_blocks=5,
           bottleneck=False, skip=True):
    """Residual convolutional network."""
    with tf.variable_scope(name):
        res = input_
        if residual_blocks != 0:
            res = conv_layer(
                input_=res, filter_size=[3, 3], dim_in=dim_in, dim_out=dim,
                name="h_in", stddev=numpy.sqrt(2. / (dim_in)),
                strides=[1, 1, 1, 1], padding="SAME",
                nonlinearity=None, bias=True,
                weight_norm=weight_norm, scale=False)
            if skip:
                out = conv_layer(
                    input_=res, filter_size=[1, 1], dim_in=dim, dim_out=dim,
                    name="skip_in", stddev=numpy.sqrt(2. / (dim)),
                    strides=[1, 1, 1, 1], padding="SAME",
                    nonlinearity=None, bias=True,
                    weight_norm=weight_norm, scale=True)

            # residual blocks
            for idx_block in xrange(residual_blocks):
                res = residual_block(res, dim, "block_%d" % idx_block,
                                     use_batch_norm=use_batch_norm, train=train,
                                     weight_norm=weight_norm,
                                     bottleneck=bottleneck)
                if skip:
                    out += conv_layer(
                        input_=res, filter_size=[1, 1], dim_in=dim, dim_out=dim,
                        name="skip_%d" % idx_block, stddev=numpy.sqrt(2. / (dim)),
                        strides=[1, 1, 1, 1], padding="SAME",
                        nonlinearity=None, bias=True,
                        weight_norm=weight_norm, scale=True)
            # outputs
            if skip:
                res = out
            if use_batch_norm:
                res = batch_norm(
                    input_=res, dim=dim, name="bn_pre_out", scale=False,
                    train=train, epsilon=1e-4, axes=[0, 1, 2])
            res = tf.nn.relu(res)
            res = conv_layer(
                input_=res, filter_size=[1, 1], dim_in=dim,
                dim_out=dim_out,
                name="out", stddev=numpy.sqrt(2. / (1. * dim)),
                strides=[1, 1, 1, 1], padding="SAME",
                nonlinearity=None, bias=True,
                weight_norm=weight_norm, scale=True)
        else:
            if bottleneck:
                res = conv_layer(
                    input_=res, filter_size=[1, 1], dim_in=dim_in, dim_out=dim,
                    name="h_0", stddev=numpy.sqrt(2. / (dim_in)),
                    strides=[1, 1, 1, 1], padding="SAME",
                    nonlinearity=None, bias=(not use_batch_norm),
                    weight_norm=weight_norm, scale=False)
                if use_batch_norm:
                    res = batch_norm(
                        input_=res, dim=dim, name="bn_0", scale=False,
                        train=train, epsilon=1e-4, axes=[0, 1, 2])
                res = tf.nn.relu(res)
                res = conv_layer(
                    input_=res, filter_size=[3, 3], dim_in=dim,
                    dim_out=dim, name="h_1", stddev=numpy.sqrt(2. / (1. * dim)),
                    strides=[1, 1, 1, 1], padding="SAME",
                    nonlinearity=None,
                    bias=(not use_batch_norm),
                    weight_norm=weight_norm, scale=False)
                if use_batch_norm:
                    res = batch_norm(
                        input_=res, dim=dim, name="bn_1", scale=False,
                        train=train, epsilon=1e-4, axes=[0, 1, 2])
                res = tf.nn.relu(res)
                res = conv_layer(
                    input_=res, filter_size=[1, 1], dim_in=dim, dim_out=dim_out,
                    name="out", stddev=numpy.sqrt(2. / (1. * dim)),
                    strides=[1, 1, 1, 1], padding="SAME",
                    nonlinearity=None, bias=True,
                    weight_norm=weight_norm, scale=True)
            else:
                res = conv_layer(
                    input_=res, filter_size=[3, 3], dim_in=dim_in, dim_out=dim,
                    name="h_0", stddev=numpy.sqrt(2. / (dim_in)),
                    strides=[1, 1, 1, 1], padding="SAME",
                    nonlinearity=None, bias=(not use_batch_norm),
                    weight_norm=weight_norm, scale=False)
                if use_batch_norm:
                    res = batch_norm(
                        input_=res, dim=dim, name="bn_0", scale=False,
                        train=train, epsilon=1e-4, axes=[0, 1, 2])
                res = tf.nn.relu(res)
                res = conv_layer(
                    input_=res, filter_size=[3, 3], dim_in=dim, dim_out=dim_out,
                    name="out", stddev=numpy.sqrt(2. / (1. * dim)),
                    strides=[1, 1, 1, 1], padding="SAME",
                    nonlinearity=None, bias=True,
                    weight_norm=weight_norm, scale=True)
        return res


# COUPLING LAYERS
# masked convolution implementations
def masked_conv_aff_coupling(input_, bitmask, mask_in, dim, name,
                             use_batch_norm=True, train=True, weight_norm=True,
                             reverse=False, residual_blocks=5,
                             bottleneck=False, use_width=1., use_height=1.,
                             mask_channel=0., skip=True):
    """Affine coupling with masked convolution."""
    with tf.variable_scope(name) as scope:
        if reverse or (not train):
            scope.reuse_variables()
        shape = input_.get_shape().as_list()
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]

        # build mask
        mask = use_width * numpy.arange(width)
        mask = use_height * numpy.arange(height).reshape((-1, 1)) + mask
        mask = mask.astype("float32")
        mask = tf.mod(mask_in + mask, 2)
        mask = tf.reshape(mask, [-1, height, width, 1])
        if mask.get_shape().as_list()[0] == 1:
            mask = tf.tile(mask, [batch_size, 1, 1, 1])
        mask = tf.mod(mask_channel + mask, 2)
        mask = tf.clip_by_value(mask + bitmask, 0., 1.)
        res = input_ * mask

        # initial input
        if use_batch_norm:
            res = batch_norm(
                input_=res, dim=channels, name="bn_in", scale=False,
                train=train, epsilon=1e-4, axes=[0, 1, 2])
            res *= 2.
        res = tf.concat([res, -res], 3)
        res = tf.concat([res, mask], 3)
        dim_in = 3. * channels
        res = tf.nn.relu(res)
        res = resnet(input_=res, dim_in=dim_in, dim=dim,
                     dim_out=2 * channels,
                     name="resnet", use_batch_norm=use_batch_norm,
                     train=train, weight_norm=weight_norm,
                     residual_blocks=residual_blocks,
                     bottleneck=bottleneck, skip=skip)
        res = tf.split(axis=3, num_or_size_splits=2, value=res)
        shift, log_rescaling = res[-2], res[-1]
        scale = variable_on_cpu(
            "rescaling_scale", [],
            tf.constant_initializer(0.))
        shift = tf.reshape(
            shift, [batch_size, height, width, channels])
        log_rescaling = tf.reshape(
            log_rescaling, [batch_size, height, width, channels])
        log_rescaling = scale * tf.tanh(log_rescaling)
        if not use_batch_norm:
            scale_shift = variable_on_cpu(
                "scale_shift", [],
                tf.constant_initializer(0.))
            log_rescaling += scale_shift
        shift *= (1. - mask)
        log_rescaling *= (1. - mask)
        if reverse:
            res = input_
            if use_batch_norm:
                mean, var = batch_norm_log_diff(
                    input_=res * (1. - mask),
                    dim=channels, name="bn_out",
                    train=False, epsilon=1e-4, axes=[0, 1, 2])
                log_var = tf.log(var)
                res *= tf.exp(.5 * log_var * (1. - mask))
                res += mean * (1. - mask)
            res *= tf.exp(-log_rescaling)
            res -= shift
            log_diff = -log_rescaling
            if use_batch_norm:
                log_diff += .5 * log_var * (1. - mask)
        else:
            res = input_
            res += shift
            res *= tf.exp(log_rescaling)
            log_diff = log_rescaling
            if use_batch_norm:
                mean, var = batch_norm_log_diff(
                    input_=res * (1. - mask),
                    dim=channels, name="bn_out",
                    train=train, epsilon=1e-4, axes=[0, 1, 2])
                log_var = tf.log(var)
                res -= mean * (1. - mask)
                res *= tf.exp(-.5 * log_var * (1. - mask))
                log_diff -= .5 * log_var * (1. - mask)

    return res, log_diff


def masked_conv_add_coupling(input_, bitmask, mask_in, dim, name,
                             use_batch_norm=True, train=True, weight_norm=True,
                             reverse=False, residual_blocks=5,
                             bottleneck=False, use_width=1., use_height=1.,
                             mask_channel=0., skip=True):
    """Additive coupling with masked convolution."""
    with tf.variable_scope(name) as scope:
        if reverse or (not train):
            scope.reuse_variables()
        shape = input_.get_shape().as_list()
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]

        # build mask
        mask = use_width * numpy.arange(width)
        mask = use_height * numpy.arange(height).reshape((-1, 1)) + mask
        mask = mask.astype("float32")
        mask = tf.mod(mask_in + mask, 2)
        mask = tf.reshape(mask, [-1, height, width, 1])
        if mask.get_shape().as_list()[0] == 1:
            mask = tf.tile(mask, [batch_size, 1, 1, 1])
        mask = tf.mod(mask_channel + mask, 2)
        mask = tf.clip_by_value(mask + bitmask, 0., 1.)
        res = input_ * mask

        # initial input
        if use_batch_norm:
            res = batch_norm(
                input_=res, dim=channels, name="bn_in", scale=False,
                train=train, epsilon=1e-4, axes=[0, 1, 2])
            res *= 2.
        res = tf.concat([res, -res], 3)
        res = tf.concat([res, mask], 3)
        dim_in = 3. * channels
        res = tf.nn.relu(res)
        shift = resnet(input_=res, dim_in=dim_in, dim=dim, dim_out=channels,
                       name="resnet", use_batch_norm=use_batch_norm,
                       train=train, weight_norm=weight_norm,
                       residual_blocks=residual_blocks,
                       bottleneck=bottleneck, skip=skip)
        shift *= (1. - mask)
        if reverse:
            res = input_
            if use_batch_norm:
                mean, var = batch_norm_log_diff(
                    input_=res * (1. - mask),
                    dim=channels, name="bn_out", train=False, epsilon=1e-4)
                log_var = tf.log(var)
                res *= tf.exp(.5 * log_var * (1. - mask))
                res += mean * (1. - mask)
            res -= shift
            log_diff = tf.zeros_like(res)
            if use_batch_norm:
                log_diff += .5 * log_var * (1. - mask)
        else:
            res = input_
            res += shift
            log_diff = tf.zeros_like(res)
            if use_batch_norm:
                mean, var = batch_norm_log_diff(
                    input_=res * (1. - mask),
                    dim=channels, name="bn_out",
                    train=train, epsilon=1e-4, axes=[0, 1, 2])
                log_var = tf.log(var)
                res -= mean * (1. - mask)
                res *= tf.exp(-.5 * log_var * (1. - mask))
                log_diff -= .5 * log_var * (1. - mask)

    return res, log_diff


def masked_conv_coupling(input_, bitmask, mask_in, dim, name,
                         use_batch_norm=True, train=True, weight_norm=True,
                         reverse=False, residual_blocks=5,
                         bottleneck=False, use_aff=True,
                         use_width=1., use_height=1.,
                         mask_channel=0., skip=True):
    """Coupling with masked convolution."""
    if use_aff:
        return masked_conv_aff_coupling(
            input_=input_, bitmask=bitmask, mask_in=mask_in, dim=dim, name=name,
            use_batch_norm=use_batch_norm, train=train, weight_norm=weight_norm,
            reverse=reverse, residual_blocks=residual_blocks,
            bottleneck=bottleneck, use_width=use_width, use_height=use_height,
            mask_channel=mask_channel, skip=skip)
    else:
        return masked_conv_add_coupling(
            input_=input_, bitmask=bitmask, mask_in=mask_in, dim=dim, name=name,
            use_batch_norm=use_batch_norm, train=train, weight_norm=weight_norm,
            reverse=reverse, residual_blocks=residual_blocks,
            bottleneck=bottleneck, use_width=use_width, use_height=use_height,
            mask_channel=mask_channel, skip=skip)


# channel-axis splitting implementations
def conv_ch_aff_coupling(input_, bitmask, dim, name,
                         use_batch_norm=True, train=True, weight_norm=True,
                         reverse=False, residual_blocks=5,
                         bottleneck=False, change_bottom=True, skip=True):
    """Affine coupling with channel-wise splitting."""
    with tf.variable_scope(name) as scope:
        if reverse or (not train):
            scope.reuse_variables()

        if change_bottom:
            input_, canvas = tf.split(
                axis=3, num_or_size_splits=2, value=input_)
            input_mask, canvas_mask = tf.split(
                axis=3, num_or_size_splits=2, value=bitmask)
        else:
            canvas, input_ = tf.split(
                axis=3, num_or_size_splits=2, value=input_)
            canvas_mask, input_mask = tf.split(
                axis=3, num_or_size_splits=2, value=bitmask)
        shape = input_.get_shape().as_list()
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]
        res = input_

        # initial input
        if use_batch_norm:
            res = batch_norm(
                input_=res, dim=channels, name="bn_in", scale=False,
                train=train, epsilon=1e-4, axes=[0, 1, 2])
        res = tf.concat([res, -res], 3)
        dim_in = 2. * channels
        res = tf.nn.relu(res)
        res = resnet(input_=res, dim_in=dim_in, dim=dim, dim_out=2 * channels,
                     name="resnet", use_batch_norm=use_batch_norm,
                     train=train, weight_norm=weight_norm,
                     residual_blocks=residual_blocks,
                     bottleneck=bottleneck, skip=skip)
        shift, log_rescaling = tf.split(
            axis=3, num_or_size_splits=2, value=res)
        scale = variable_on_cpu(
            "scale", [],
            tf.constant_initializer(1.))
        shift = tf.reshape(
            shift, [batch_size, height, width, channels])
        log_rescaling = tf.reshape(
            log_rescaling, [batch_size, height, width, channels])
        log_rescaling = scale * tf.tanh(log_rescaling)
        if not use_batch_norm:
            scale_shift = variable_on_cpu(
                "scale_shift", [],
                tf.constant_initializer(0.))
            log_rescaling += scale_shift
        shift *= (1. - canvas_mask)
        log_rescaling *= (1. - canvas_mask)
        if reverse:
            res = canvas
            if use_batch_norm:
                mean, var = batch_norm_log_diff(
                    input_=res * (1. - canvas_mask),
                    dim=channels, name="bn_out", train=False,
                    epsilon=1e-4, axes=[0, 1, 2])
                log_var = tf.log(var)
                res *= tf.exp(.5 * log_var * (1. - canvas_mask))
                res += mean * (1. - canvas_mask)
            res *= tf.exp(-log_rescaling)
            res -= shift
            log_diff = -log_rescaling
            if use_batch_norm:
                log_diff += .5 * log_var * (1. - canvas_mask)
        else:
            res = canvas
            res += shift
            res *= tf.exp(log_rescaling)
            log_diff = log_rescaling
            if use_batch_norm:
                mean, var = batch_norm_log_diff(
                    input_=res * (1. - canvas_mask),
                    dim=channels, name="bn_out", train=train,
                    epsilon=1e-4, axes=[0, 1, 2])
                log_var = tf.log(var)
                res -= mean * (1. - canvas_mask)
                res *= tf.exp(-.5 * log_var * (1. - canvas_mask))
                log_diff -= .5 * log_var * (1. - canvas_mask)
        if change_bottom:
            res = tf.concat([input_, res], 3)
            log_diff = tf.concat([tf.zeros_like(log_diff), log_diff], 3)
        else:
            res = tf.concat([res, input_], 3)
            log_diff = tf.concat([log_diff, tf.zeros_like(log_diff)], 3)

    return res, log_diff


def conv_ch_add_coupling(input_, bitmask, dim, name,
                         use_batch_norm=True, train=True, weight_norm=True,
                         reverse=False, residual_blocks=5,
                         bottleneck=False, change_bottom=True, skip=True):
    """Additive coupling with channel-wise splitting."""
    with tf.variable_scope(name) as scope:
        if reverse or (not train):
            scope.reuse_variables()

        if change_bottom:
            input_, canvas = tf.split(
                axis=3, num_or_size_splits=2, value=input_)
            input_mask, canvas_mask = tf.split(
                axis=3, num_or_size_splits=2, value=bitmask)
        else:
            canvas, input_ = tf.split(
                axis=3, num_or_size_splits=2, value=input_)
            canvas_mask, input_mask = tf.split(
                axis=3, num_or_size_splits=2, value=bitmask)
        shape = input_.get_shape().as_list()
        channels = shape[3]
        res = input_

        # initial input
        if use_batch_norm:
            res = batch_norm(
                input_=res, dim=channels, name="bn_in", scale=False,
                train=train, epsilon=1e-4, axes=[0, 1, 2])
        res = tf.concat([res, -res], 3)
        dim_in = 2. * channels
        res = tf.nn.relu(res)
        shift = resnet(input_=res, dim_in=dim_in, dim=dim, dim_out=channels,
                       name="resnet", use_batch_norm=use_batch_norm,
                       train=train, weight_norm=weight_norm,
                       residual_blocks=residual_blocks,
                       bottleneck=bottleneck, skip=skip)
        shift *= (1. - canvas_mask)
        if reverse:
            res = canvas
            if use_batch_norm:
                mean, var = batch_norm_log_diff(
                    input_=res * (1. - canvas_mask),
                    dim=channels, name="bn_out", train=False,
                    epsilon=1e-4, axes=[0, 1, 2])
                log_var = tf.log(var)
                res *= tf.exp(.5 * log_var * (1. - canvas_mask))
                res += mean * (1. - canvas_mask)
            res -= shift
            log_diff = tf.zeros_like(res)
            if use_batch_norm:
                log_diff += .5 * log_var * (1. - canvas_mask)
        else:
            res = canvas
            res += shift
            log_diff = tf.zeros_like(res)
            if use_batch_norm:
                mean, var = batch_norm_log_diff(
                    input_=res * (1. - canvas_mask),
                    dim=channels, name="bn_out", train=train,
                    epsilon=1e-4, axes=[0, 1, 2])
                log_var = tf.log(var)
                res -= mean * (1. - canvas_mask)
                res *= tf.exp(-.5 * log_var * (1. - canvas_mask))
                log_diff -= .5 * log_var * (1. - canvas_mask)
        if change_bottom:
            res = tf.concat([input_, res], 3)
            log_diff = tf.concat([tf.zeros_like(log_diff), log_diff], 3)
        else:
            res = tf.concat([res, input_], 3)
            log_diff = tf.concat([log_diff, tf.zeros_like(log_diff)], 3)

    return res, log_diff


def conv_ch_coupling(input_, bitmask, dim, name,
                     use_batch_norm=True, train=True, weight_norm=True,
                     reverse=False, residual_blocks=5,
                     bottleneck=False, use_aff=True, change_bottom=True,
                     skip=True):
    """Coupling with channel-wise splitting."""
    if use_aff:
        return conv_ch_aff_coupling(
            input_=input_, bitmask=bitmask, dim=dim, name=name,
            use_batch_norm=use_batch_norm, train=train, weight_norm=weight_norm,
            reverse=reverse, residual_blocks=residual_blocks,
            bottleneck=bottleneck, change_bottom=change_bottom, skip=skip)
    else:
        return conv_ch_add_coupling(
            input_=input_, bitmask=bitmask, dim=dim, name=name,
            use_batch_norm=use_batch_norm, train=train, weight_norm=weight_norm,
            reverse=reverse, residual_blocks=residual_blocks,
            bottleneck=bottleneck, change_bottom=change_bottom, skip=skip)


# RECURSIVE USE OF COUPLING LAYERS
def rec_masked_conv_coupling(input_, bitmask, hps, scale_idx, n_scale,
                             use_batch_norm=True, weight_norm=True,
                             train=True):
    """Recursion on coupling layers."""
    shape = input_.get_shape().as_list()
    channels = shape[3]
    residual_blocks = hps.residual_blocks
    base_dim = hps.base_dim
    mask = 1.
    use_aff = hps.use_aff
    res = input_
    skip = hps.skip
    log_diff = tf.zeros_like(input_)
    dim = base_dim
    if _recursion_type < 4:
        dim *= 2 ** scale_idx
    with tf.variable_scope("scale_%d" % scale_idx):
        # initial coupling layers
        res, inc_log_diff = masked_conv_coupling(
            input_=res, bitmask=bitmask,
            mask_in=mask, dim=dim,
            name="coupling_0",
            use_batch_norm=use_batch_norm, train=train,
            weight_norm=weight_norm,
            reverse=False, residual_blocks=residual_blocks,
            bottleneck=hps.bottleneck, use_aff=use_aff,
            use_width=1., use_height=1., skip=skip)
        log_diff += inc_log_diff
        res, inc_log_diff = masked_conv_coupling(
            input_=res, bitmask=bitmask,
            mask_in=1. - mask, dim=dim,
            name="coupling_1",
            use_batch_norm=use_batch_norm, train=train,
            weight_norm=weight_norm,
            reverse=False, residual_blocks=residual_blocks,
            bottleneck=hps.bottleneck, use_aff=use_aff,
            use_width=1., use_height=1., skip=skip)
        log_diff += inc_log_diff
        res, inc_log_diff = masked_conv_coupling(
            input_=res, bitmask=bitmask,
            mask_in=mask, dim=dim,
            name="coupling_2",
            use_batch_norm=use_batch_norm, train=train,
            weight_norm=weight_norm,
            reverse=False, residual_blocks=residual_blocks,
            bottleneck=hps.bottleneck, use_aff=True,
            use_width=1., use_height=1., skip=skip)
        log_diff += inc_log_diff
    if scale_idx < (n_scale - 1):
        with tf.variable_scope("scale_%d" % scale_idx):
            res = squeeze_2x2(res)
            bitmask = squeeze_2x2(bitmask)
            log_diff = squeeze_2x2(log_diff)
            res, inc_log_diff = conv_ch_coupling(
                input_=res, bitmask=bitmask,
                change_bottom=True, dim=2 * dim,
                name="coupling_4",
                use_batch_norm=use_batch_norm, train=train,
                weight_norm=weight_norm,
                reverse=False, residual_blocks=residual_blocks,
                bottleneck=hps.bottleneck, use_aff=use_aff, skip=skip)
            log_diff += inc_log_diff
            res, inc_log_diff = conv_ch_coupling(
                input_=res, bitmask=bitmask,
                change_bottom=False, dim=2 * dim,
                name="coupling_5",
                use_batch_norm=use_batch_norm, train=train,
                weight_norm=weight_norm,
                reverse=False, residual_blocks=residual_blocks,
                bottleneck=hps.bottleneck, use_aff=use_aff, skip=skip)
            log_diff += inc_log_diff
            res, inc_log_diff = conv_ch_coupling(
                input_=res, bitmask=bitmask,
                change_bottom=True, dim=2 * dim,
                name="coupling_6",
                use_batch_norm=use_batch_norm, train=train,
                weight_norm=weight_norm,
                reverse=False, residual_blocks=residual_blocks,
                bottleneck=hps.bottleneck, use_aff=True, skip=skip)
            log_diff += inc_log_diff
            res = unsqueeze_2x2(res)
            bitmask = unsqueeze_2x2(bitmask)
            log_diff = unsqueeze_2x2(log_diff)
        if _recursion_type > 1:
            res = squeeze_2x2_ordered(res)
            bitmask = squeeze_2x2_ordered(bitmask)
            log_diff = squeeze_2x2_ordered(log_diff)
            if _recursion_type > 2:
                res_1 = res[:, :, :, :channels]
                res_2 = res[:, :, :, channels:]
                bitmask_1 = bitmask[:, :, :, :channels]
                bitmask_2 = bitmask[:, :, :, channels:]
                log_diff_1 = log_diff[:, :, :, :channels]
                log_diff_2 = log_diff[:, :, :, channels:]
            else:
                res_1, res_2 = tf.split(
                    axis=3, num_or_size_splits=2, value=res)
                bitmask_1, bitmask_2 = tf.split(
                    axis=3, num_or_size_splits=2, value=bitmask)
                log_diff_1, log_diff_2 = tf.split(
                    axis=3, num_or_size_splits=2, value=log_diff)
            res_1, inc_log_diff = rec_masked_conv_coupling(
                input_=res_1, bitmask=bitmask_1, hps=hps, scale_idx=scale_idx + 1,
                n_scale=n_scale, use_batch_norm=use_batch_norm, weight_norm=weight_norm,
                train=train)
            res = tf.concat([res_1, res_2], 3)
            bitmask = tf.concat([bitmask_1, bitmask_2], 3)
            log_diff_1 += inc_log_diff
            log_diff = tf.concat([log_diff_1, log_diff_2], 3)
            res = squeeze_2x2_ordered(res, reverse=True)
            bitmask = squeeze_2x2_ordered(bitmask, reverse=True)
            log_diff = squeeze_2x2_ordered(log_diff, reverse=True)
        else:
            res = squeeze_2x2_ordered(res)
            bitmask = squeeze_2x2_ordered(bitmask)
            log_diff = squeeze_2x2_ordered(log_diff)
            res, inc_log_diff = rec_masked_conv_coupling(
                input_=res, bitmask=bitmask, hps=hps, scale_idx=scale_idx + 1,
                n_scale=n_scale, use_batch_norm=use_batch_norm, weight_norm=weight_norm,
                train=train)
            log_diff += inc_log_diff
            res = squeeze_2x2_ordered(res, reverse=True)
            bitmask = squeeze_2x2_ordered(bitmask, reverse=True)
            log_diff = squeeze_2x2_ordered(log_diff, reverse=True)
    else:
        with tf.variable_scope("scale_%d" % scale_idx):
            res, inc_log_diff = masked_conv_coupling(
                input_=res, bitmask=bitmask,
                mask_in=1. - mask, dim=dim,
                name="coupling_3",
                use_batch_norm=use_batch_norm, train=train,
                weight_norm=weight_norm,
                reverse=False, residual_blocks=residual_blocks,
                bottleneck=hps.bottleneck, use_aff=True,
                use_width=1., use_height=1., skip=skip)
            log_diff += inc_log_diff
    return res, log_diff


def rec_masked_deconv_coupling(input_, bitmask, hps, scale_idx, n_scale,
                               use_batch_norm=True, weight_norm=True,
                               train=True):
    """Recursion on inverting coupling layers."""
    shape = input_.get_shape().as_list()
    channels = shape[3]
    residual_blocks = hps.residual_blocks
    base_dim = hps.base_dim
    mask = 1.
    use_aff = hps.use_aff
    res = input_
    log_diff = tf.zeros_like(input_)
    skip = hps.skip
    dim = base_dim
    if _recursion_type < 4:
        dim *= 2 ** scale_idx
    if scale_idx < (n_scale - 1):
        if _recursion_type > 1:
            res = squeeze_2x2_ordered(res)
            bitmask = squeeze_2x2_ordered(bitmask)
            log_diff = squeeze_2x2_ordered(log_diff)
            if _recursion_type > 2:
                res_1 = res[:, :, :, :channels]
                res_2 = res[:, :, :, channels:]
                bitmask_1 = bitmask[:, :, :, :channels]
                bitmask_2 = bitmask[:, :, :, channels:]
                log_diff_1 = log_diff[:, :, :, :channels]
                log_diff_2 = log_diff[:, :, :, channels:]
            else:
                res_1, res_2 = tf.split(
                    axis=3, num_or_size_splits=2, value=res)
                bitmask_1, bitmask_2 = tf.split(
                    axis=3, num_or_size_splits=2, value=bitmask)
                log_diff_1, log_diff_2 = tf.split(
                    axis=3, num_or_size_splits=2, value=log_diff)
            res_1, log_diff_1 = rec_masked_deconv_coupling(
                input_=res_1, bitmask=bitmask_1, hps=hps,
                scale_idx=scale_idx + 1, n_scale=n_scale,
                use_batch_norm=use_batch_norm, weight_norm=weight_norm,
                train=train)
            res = tf.concat([res_1, res_2], 3)
            bitmask = tf.concat([bitmask_1, bitmask_2], 3)
            log_diff = tf.concat([log_diff_1, log_diff_2], 3)
            res = squeeze_2x2_ordered(res, reverse=True)
            bitmask = squeeze_2x2_ordered(bitmask, reverse=True)
            log_diff = squeeze_2x2_ordered(log_diff, reverse=True)
        else:
            res = squeeze_2x2_ordered(res)
            bitmask = squeeze_2x2_ordered(bitmask)
            log_diff = squeeze_2x2_ordered(log_diff)
            res, log_diff = rec_masked_deconv_coupling(
                input_=res, bitmask=bitmask, hps=hps,
                scale_idx=scale_idx + 1, n_scale=n_scale,
                use_batch_norm=use_batch_norm, weight_norm=weight_norm,
                train=train)
            res = squeeze_2x2_ordered(res, reverse=True)
            bitmask = squeeze_2x2_ordered(bitmask, reverse=True)
            log_diff = squeeze_2x2_ordered(log_diff, reverse=True)
        with tf.variable_scope("scale_%d" % scale_idx):
            res = squeeze_2x2(res)
            bitmask = squeeze_2x2(bitmask)
            log_diff = squeeze_2x2(log_diff)
            res, inc_log_diff = conv_ch_coupling(
                input_=res, bitmask=bitmask,
                change_bottom=True, dim=2 * dim,
                name="coupling_6",
                use_batch_norm=use_batch_norm, train=train,
                weight_norm=weight_norm,
                reverse=True, residual_blocks=residual_blocks,
                bottleneck=hps.bottleneck, use_aff=True, skip=skip)
            log_diff += inc_log_diff
            res, inc_log_diff = conv_ch_coupling(
                input_=res, bitmask=bitmask,
                change_bottom=False, dim=2 * dim,
                name="coupling_5",
                use_batch_norm=use_batch_norm, train=train,
                weight_norm=weight_norm,
                reverse=True, residual_blocks=residual_blocks,
                bottleneck=hps.bottleneck, use_aff=use_aff, skip=skip)
            log_diff += inc_log_diff
            res, inc_log_diff = conv_ch_coupling(
                input_=res, bitmask=bitmask,
                change_bottom=True, dim=2 * dim,
                name="coupling_4",
                use_batch_norm=use_batch_norm, train=train,
                weight_norm=weight_norm,
                reverse=True, residual_blocks=residual_blocks,
                bottleneck=hps.bottleneck, use_aff=use_aff, skip=skip)
            log_diff += inc_log_diff
            res = unsqueeze_2x2(res)
            bitmask = unsqueeze_2x2(bitmask)
            log_diff = unsqueeze_2x2(log_diff)
    else:
        with tf.variable_scope("scale_%d" % scale_idx):
            res, inc_log_diff = masked_conv_coupling(
                input_=res, bitmask=bitmask,
                mask_in=1. - mask, dim=dim,
                name="coupling_3",
                use_batch_norm=use_batch_norm, train=train,
                weight_norm=weight_norm,
                reverse=True, residual_blocks=residual_blocks,
                bottleneck=hps.bottleneck, use_aff=True,
                use_width=1., use_height=1., skip=skip)
            log_diff += inc_log_diff

    with tf.variable_scope("scale_%d" % scale_idx):
        res, inc_log_diff = masked_conv_coupling(
            input_=res, bitmask=bitmask,
            mask_in=mask, dim=dim,
            name="coupling_2",
            use_batch_norm=use_batch_norm, train=train, weight_norm=weight_norm,
            reverse=True, residual_blocks=residual_blocks,
            bottleneck=hps.bottleneck, use_aff=True,
            use_width=1., use_height=1., skip=skip)
        log_diff += inc_log_diff
        res, inc_log_diff = masked_conv_coupling(
            input_=res, bitmask=bitmask,
            mask_in=1. - mask, dim=dim,
            name="coupling_1",
            use_batch_norm=use_batch_norm, train=train, weight_norm=weight_norm,
            reverse=True, residual_blocks=residual_blocks,
            bottleneck=hps.bottleneck, use_aff=use_aff,
            use_width=1., use_height=1., skip=skip)
        log_diff += inc_log_diff
        res, inc_log_diff = masked_conv_coupling(
            input_=res, bitmask=bitmask,
            mask_in=mask, dim=dim,
            name="coupling_0",
            use_batch_norm=use_batch_norm, train=train, weight_norm=weight_norm,
            reverse=True, residual_blocks=residual_blocks,
            bottleneck=hps.bottleneck, use_aff=use_aff,
            use_width=1., use_height=1., skip=skip)
        log_diff += inc_log_diff

    return res, log_diff


# ENCODER AND DECODER IMPLEMENTATIONS
# start the recursions
def encoder_spec(input_, bitmask, hps, n_scale, use_batch_norm=True,
                 weight_norm=True, train=True):
    """Encoding/gaussianization function."""
    res = input_
    log_diff = tf.zeros_like(input_)
    res, inc_log_diff = rec_masked_conv_coupling(
        input_=res, bitmask=bitmask, hps=hps, scale_idx=0, n_scale=n_scale,
        use_batch_norm=use_batch_norm, weight_norm=weight_norm,
        train=train)
    log_diff += inc_log_diff

    return res, log_diff


def decoder_spec(input_, bitmask, hps, n_scale, use_batch_norm=True,
                 weight_norm=True, train=True):
    """Decoding/generator function."""
    res, log_diff = rec_masked_deconv_coupling(
        input_=input_, bitmask=bitmask, hps=hps, scale_idx=0, n_scale=n_scale,
        use_batch_norm=use_batch_norm, weight_norm=weight_norm,
        train=train)

    return res, log_diff
