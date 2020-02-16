import tensorflow as tf
import numpy as np
image_grid = tf.contrib.gan.eval.image_grid

from .rnvp.modules import encoder_spec, decoder_spec
from .rnvp.utils import standard_normal_ll, standard_normal_sample
from .rnvp.logits import preprocess, postprocess


def rearrange(image_tensor):
    B, H, W, C = image_tensor.get_shape().as_list()
    L = int(np.sqrt(B))
    img = image_grid(image_tensor[:L * L], [L, L], [H, W], C)

    return img


class Model(object):
    def __init__(self, hps):
        self.hps = hps

    def forward(self, x, m, train):
        '''
        Args:
            x: data, [B,H,W,C] [uint8]
            m: mask, [B,H,W,C] [uint8]
        '''
        reverse = False
        m = tf.cast(m, tf.float32)
        x, logdet = preprocess(x, m, self.hps.data_constraint, self.hps.n_bits)
        z, ldet = encoder_spec(x, m, self.hps, self.hps.n_scale,
                               use_batch_norm=self.hps.use_batch_norm,
                               train=train)
        ldet = tf.reduce_sum(ldet, [1, 2, 3])
        logdet += ldet
        prior_ll = standard_normal_ll(z)
        prior_ll = tf.reduce_sum(prior_ll * (1. - m), [1, 2, 3])
        log_likel = prior_ll + logdet

        return log_likel

    def inverse(self, x, m, train):
        reverse = True
        m = tf.cast(m, tf.float32)
        x, _ = preprocess(x, m, self.hps.data_constraint, self.hps.n_bits)
        m = tf.expand_dims(m, axis=1)
        m = tf.tile(m, [1, self.hps.num_samples, 1, 1, 1])
        z = standard_normal_sample(
            [self.hps.batch_size, self.hps.num_samples] + self.hps.image_shape)
        z = z * self.hps.sample_std
        x = tf.expand_dims(x, axis=1)
        x = z * (1. - m) + x * m
        x = tf.reshape(
            x, [self.hps.batch_size * self.hps.num_samples] + self.hps.image_shape)
        m = tf.reshape(
            m, [self.hps.batch_size * self.hps.num_samples] + self.hps.image_shape)
        x, _ = decoder_spec(x, m, self.hps, self.hps.n_scale,
                            use_batch_norm=self.hps.use_batch_norm,
                            train=train)
        x, _ = postprocess(x, m, self.hps.data_constraint, self.hps.n_bits)
        x = tf.reshape(
            x, [self.hps.batch_size, self.hps.num_samples] + self.hps.image_shape)
        return x

    def inverse_zero(self, x, m, train):
        reverse = True
        m = tf.cast(m, tf.float32)
        z = tf.zeros([self.hps.batch_size] +
                     self.hps.image_shape, dtype=tf.float32)
        x, _ = preprocess(x, m, self.hps.data_constraint, self.hps.n_bits)
        x = z * (1. - m) + x * m
        x, _ = decoder_spec(x, m, self.hps, self.hps.n_scale,
                            use_batch_norm=self.hps.use_batch_norm,
                            train=train)
        x, _ = postprocess(x, m, self.hps.data_constraint, self.hps.n_bits)
        return x

    def build(self, trainset, validset, testset):
        train_x, train_m = trainset.x, trainset.m
        valid_x, valid_m = validset.x, validset.m
        test_x, test_m = testset.x, testset.m
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            self.train_ll = self.forward(train_x, train_m, True)
            self.valid_ll = self.forward(valid_x, valid_m, False)
            self.test_ll = self.forward(test_x, test_m, False)
            self.train_sam = self.inverse(train_x, train_m, False)
            self.valid_sam = self.inverse(valid_x, valid_m, False)
            self.test_sam = self.inverse(test_x, test_m, False)
            self.train_sam_mean = self.inverse_zero(train_x, train_m, False)
            self.valid_sam_mean = self.inverse_zero(valid_x, valid_m, False)
            self.test_sam_mean = self.inverse_zero(test_x, test_m, False)
            # image summ
            gray_img = tf.ones_like(train_x) * 128
            train_gt = rearrange(train_x)
            tf.summary.image('train/gt', train_gt)
            train_in = rearrange(train_x * train_m + gray_img * (1 - train_m))
            tf.summary.image('train/in', train_in)
            for i in range(self.hps.num_samples):
                train_out = rearrange(self.train_sam[:, i])
                tf.summary.image(f'train/out_{i}', train_out)

            # loss
            nll = tf.reduce_mean(-self.train_ll)
            tf.summary.scalar('nll', nll)
            l2_reg = sum(
                [tf.reduce_sum(tf.square(v)) for v in tf.trainable_variables()
                 if ("magnitude" in v.name) or ("rescaling_scale" in v.name)])
            loss = nll + self.hps.lambda_reg * l2_reg
            if self.hps.lambda_mse > 0:
                mse_loss = tf.square(
                    (self.train_sam_mean - tf.cast(train_x, tf.float32)) / 255.)
                mse_loss = tf.reduce_sum(mse_loss, [1, 2, 3])
                mse_loss = tf.reduce_mean(mse_loss)
                tf.summary.scalar('mse_loss', mse_loss)
                loss += mse_loss * self.hps.lambda_mse
            tf.summary.scalar('loss', loss)

            # train
            self.global_step = tf.train.get_or_create_global_step()
            learning_rate = tf.train.inverse_time_decay(
                self.hps.lr, self.global_step,
                self.hps.decay_steps, self.hps.decay_rate,
                staircase=True)
            tf.summary.scalar('lr', learning_rate)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=0.9, beta2=0.999, epsilon=1e-08,
                use_locking=False, name="Adam")
            grads_and_vars = optimizer.compute_gradients(
                loss, tf.trainable_variables())
            grads, vars_ = zip(*grads_and_vars)
            if self.hps.clip_gradient > 0:
                grads, gradient_norm = tf.clip_by_global_norm(
                    grads, clip_norm=self.hps.clip_gradient)
                gradient_norm = tf.check_numerics(
                    gradient_norm, "Gradient norm is NaN or Inf.")
                tf.summary.scalar('gradient_norm', gradient_norm)
            capped_grads_and_vars = zip(grads, vars_)
            self.train_op = optimizer.apply_gradients(
                capped_grads_and_vars, global_step=self.global_step)

            # summary
            self.summ_op = tf.summary.merge_all()
