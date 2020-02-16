"""
The core Pixel-CNN model
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
from . import nn


def pixelcnn_spec(x, h=None, hparams=None, init=False, dropout_p=0.5, ema=None):
    """
    We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
    a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
    of the x_out tensor describes the predictive distribution for the RGB at
    that position.
    'h' is an optional N x K matrix of values to condition our generative model on
    """

    counters = {}
    with arg_scope([nn.conv2d, nn.deconv2d, nn.gated_resnet, nn.dense], counters=counters, init=init, ema=ema, dropout_p=dropout_p):

        # parse resnet nonlinearity argument
        if hparams.resnet_nonlinearity == 'concat_elu':
            resnet_nonlinearity = nn.concat_elu
        elif hparams.resnet_nonlinearity == 'elu':
            resnet_nonlinearity = tf.nn.elu
        elif hparams.resnet_nonlinearity == 'relu':
            resnet_nonlinearity = tf.nn.relu
        else:
            raise('resnet nonlinearity ' +
                  hparams.resnet_nonlinearity + ' is not supported')

        with arg_scope([nn.gated_resnet], nonlinearity=resnet_nonlinearity, h=h):

            # ////////// up pass through pixelCNN ////////
            xs = nn.int_shape(x)
            # add channel of ones to distinguish image from padding later on
            x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], 3)
            u_list = [nn.down_shift(nn.down_shifted_conv2d(
                x_pad, num_filters=hparams.nr_filters, filter_size=[2, 3]))]  # stream for pixels above
            ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=hparams.nr_filters, filter_size=[1, 3])) +
                       nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=hparams.nr_filters, filter_size=[2, 1]))]  # stream for up and to the left

            for rep in range(hparams.nr_resnet):
                u_list.append(nn.gated_resnet(
                    u_list[-1], conv=nn.down_shifted_conv2d))
                ul_list.append(nn.gated_resnet(
                    ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))

            u_list.append(nn.down_shifted_conv2d(
                u_list[-1], num_filters=hparams.nr_filters, stride=[2, 2]))
            ul_list.append(nn.down_right_shifted_conv2d(
                ul_list[-1], num_filters=hparams.nr_filters, stride=[2, 2]))

            for rep in range(hparams.nr_resnet):
                u_list.append(nn.gated_resnet(
                    u_list[-1], conv=nn.down_shifted_conv2d))
                ul_list.append(nn.gated_resnet(
                    ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))

            u_list.append(nn.down_shifted_conv2d(
                u_list[-1], num_filters=hparams.nr_filters, stride=[2, 2]))
            ul_list.append(nn.down_right_shifted_conv2d(
                ul_list[-1], num_filters=hparams.nr_filters, stride=[2, 2]))

            for rep in range(hparams.nr_resnet):
                u_list.append(nn.gated_resnet(
                    u_list[-1], conv=nn.down_shifted_conv2d))
                ul_list.append(nn.gated_resnet(
                    ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))

            # remember nodes
            for t in u_list + ul_list:
                tf.add_to_collection('checkpoints', t)

            # /////// down pass ////////
            u = u_list.pop()
            ul = ul_list.pop()
            for rep in range(hparams.nr_resnet):
                u = nn.gated_resnet(
                    u, u_list.pop(), conv=nn.down_shifted_conv2d)
                ul = nn.gated_resnet(ul, tf.concat(
                    [u, ul_list.pop()], 3), conv=nn.down_right_shifted_conv2d)
                tf.add_to_collection('checkpoints', u)
                tf.add_to_collection('checkpoints', ul)

            u = nn.down_shifted_deconv2d(
                u, num_filters=hparams.nr_filters, stride=[2, 2])
            ul = nn.down_right_shifted_deconv2d(
                ul, num_filters=hparams.nr_filters, stride=[2, 2])

            for rep in range(hparams.nr_resnet + 1):
                u = nn.gated_resnet(
                    u, u_list.pop(), conv=nn.down_shifted_conv2d)
                ul = nn.gated_resnet(ul, tf.concat(
                    [u, ul_list.pop()], 3), conv=nn.down_right_shifted_conv2d)
                tf.add_to_collection('checkpoints', u)
                tf.add_to_collection('checkpoints', ul)

            u = nn.down_shifted_deconv2d(
                u, num_filters=hparams.nr_filters, stride=[2, 2])
            ul = nn.down_right_shifted_deconv2d(
                ul, num_filters=hparams.nr_filters, stride=[2, 2])

            for rep in range(hparams.nr_resnet + 1):
                u = nn.gated_resnet(
                    u, u_list.pop(), conv=nn.down_shifted_conv2d)
                ul = nn.gated_resnet(ul, tf.concat(
                    [u, ul_list.pop()], 3), conv=nn.down_right_shifted_conv2d)
                tf.add_to_collection('checkpoints', u)
                tf.add_to_collection('checkpoints', ul)

            C = hparams.image_shape[-1]
            K = C * 2 + C * (C - 1) + 1
            x_out = nn.nin(tf.nn.elu(ul), K * hparams.nr_mixture)

            assert len(u_list) == 0
            assert len(ul_list) == 0

            return x_out
