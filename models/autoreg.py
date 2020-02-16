import tensorflow as tf
import numpy as np
image_grid = tf.contrib.gan.eval.image_grid

from .rnvp.modules import resnet
from .rnvp.logits import preprocess, postprocess
from .pixelcnn.pixelcnn import pixelcnn_spec
from .pixelcnn.mixture import mixture_likelihoods, sample_mixture

class Model(object):
    def __init__(self, hps):
        self.hps = hps

        self.pixelcnn = tf.make_template('PixelCNN', pixelcnn_spec)

    def embed(self, x, m, train):
        x = x * m
        res = tf.concat([x, m], axis=-1)
        with tf.variable_scope('embedding'):
            for i in range(self.hps.n_scale):
                dim_in = int(res.get_shape()[-1])
                dim_out = dim_in * 2
                res = resnet(res, dim_in, self.hps.base_dim,
                            dim_out, name='l{}'.format(i),
                            use_batch_norm=False, train=train,
                            residual_blocks=self.hps.residual_blocks,
                            bottleneck=self.hps.bottleneck, skip=self.hps.skip)
                res = tf.layers.max_pooling2d(res, 2, 2)
            res = tf.layers.flatten(res)
            f = tf.layers.dense(res, self.hps.embed_dim, name='feat')

        return f

    def forward(self, x, m, h, logdet, train, init=False):
        dropout = self.hps.dropout_rate if train else 0.0
        inp = tf.concat([x, m], axis=-1)

        likel_param = self.pixelcnn(
            inp, h=h, hparams=self.hps,
            init=init, dropout_p=dropout)
        ll = mixture_likelihoods(likel_param, x, self.hps.image_shape[-1])
        ll = tf.reduce_sum(ll * (1. - m), [1, 2, 3])
        ll = ll + logdet

        return ll

    def inverse(self, x, m, h, s):
        inp = tf.concat([x, m], axis=-1)

        likel_param = self.pixelcnn(
            inp, h=h, hparams=self.hps,
            init=False, dropout_p=0.)
        
        sam = sample_mixture(likel_param, self.hps.image_shape[-1])
        mask = (1 - m) * s
        sam = sam * mask + x * (1 - mask)
        sam, _ = postprocess(sam, m)

        return sam

    def build(self, trainset, validset, testset):
        train_x, train_m = trainset.x, trainset.m
        valid_x, valid_m = validset.x, validset.m
        test_x, test_m = testset.x, testset.m
        
        train_m = tf.cast(train_m, tf.float32)
        train_x, train_ldet = preprocess(train_x, train_m)
        valid_m = tf.cast(valid_m, tf.float32)
        valid_x, valid_ldet = preprocess(valid_x, valid_m)
        test_m = tf.cast(test_m, tf.float32)
        test_x, test_ldet = preprocess(test_x, test_m)

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            init_shape = [self.hps.batch_size * self.hps.init_batches] + self.hps.image_shape
            self.init_x = tf.placeholder(tf.uint8, init_shape)
            self.init_m = tf.placeholder(tf.uint8, init_shape)
            init_m = tf.cast(self.init_m, tf.float32)
            init_x, init_ldet = preprocess(self.init_x, init_m)
            init_h = self.embed(init_x, init_m, True)
            self.init = self.forward(init_x, init_m, init_h, init_ldet, True, True)

            train_h = self.embed(train_x, train_m, True)
            valid_h = self.embed(valid_x, valid_m, False)
            test_h = self.embed(test_x, test_m, False)

            self.train_ll = self.forward(train_x, train_m, train_h, train_ldet, True)
            self.valid_ll = self.forward(valid_x, valid_m, valid_h, valid_ldet, False)
            self.test_ll = self.forward(test_x, test_m, test_h, test_ldet, False)

            # sample
            _shape = [self.hps.batch_size] + self.hps.image_shape
            self.x_ph = tf.placeholder(tf.float32, _shape)
            self.m_ph = tf.placeholder(tf.float32, _shape)
            self.s_ph = tf.placeholder(tf.float32, _shape)
            x_ph, _ = preprocess(self.x_ph, self.m_ph, rand=False)
            h_ph = self.embed(x_ph, self.m_ph, False)
            self.sam = self.inverse(x_ph, self.m_ph, h_ph, self.s_ph)

            # loss
            nll = tf.reduce_mean(-self.train_ll)
            tf.summary.scalar('nll', nll)
            l2_reg = sum(
                [tf.reduce_sum(tf.square(v)) for v in tf.trainable_variables()
                 if ("magnitude" in v.name) or ("rescaling_scale" in v.name)])
            loss = nll + self.hps.lambda_reg * l2_reg
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

    def sample_once(self, sess, x_in, m_in):
        B, H, W, C = _shape = [self.hps.batch_size] + self.hps.image_shape
        x_nda = x_in.astype(np.float32) + np.random.uniform(size=x_in.shape)
        m_nda = m_in.astype(np.float32)
        s_nda = np.zeros_like(m_nda)
        for yi in range(H):
            for xi in range(W):
                s_nda[:, yi, xi, :] = 1.
                feed_dict = {
                    self.x_ph: x_nda,
                    self.m_ph: m_nda,
                    self.s_ph: s_nda
                }
                new_x = sess.run(self.sam, feed_dict)
                x_nda[:, yi, xi, :] = new_x[:, yi, xi, :]
                s_nda[:, yi, xi, :] = 0.
        x_nda = np.clip(x_nda, 0, 255)

        return x_nda.astype(np.uint8)

    def sample(self, sess, x_in, m_in):
        sams = []
        for i in range(self.hps.num_samples):
            sam = self.sample_once(sess, x_in, m_in)
            sams.append(sam)
        sams = np.stack(sams, axis=1)

        return sams
        










