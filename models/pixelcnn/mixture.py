import tensorflow as tf
import numpy as np


def mixture_likelihoods_3c(params, targets, base_distribution='gaussian',
                           name='mixture_likelihood'):
    """Given log-unnormalized mixture weights, shift, and log scale parameters
    for mixture components, return the likelihoods for targets.
    Args:
        params: N x H x W x 10*ncomp tensor of parameters of mixture model
        targets: N x H x W x 3 tensor of 2d targets to get likelihoods for.
        base_distribution: {'gaussian', 'laplace', 'logistic'} the base
            distribution of mixture components.
    Return:
        likelihoods: N x H x W x3  tensor of likelihoods.
    """
    base_distribution = base_distribution.lower()
    xs = targets.get_shape().as_list()
    ps = params.get_shape().as_list()
    nr_mix = ps[-1] // 10
    with tf.variable_scope(name):
        # Compute likelihoods per target and component
        # Write log likelihood as logsumexp.
        targets = tf.reshape(targets, xs + [1])
        logits = tf.reshape(params[:, :, :, :nr_mix], xs[:-1] + [1, nr_mix])
        params = tf.reshape(params[:, :, :, nr_mix:], xs + [3 * nr_mix])
        means, lsigmas, coeffs = tf.split(params, 3, 4)
        sigmas = tf.exp(lsigmas)
        coeffs = tf.nn.tanh(coeffs)
        m0 = tf.reshape(means[:, :, :, 0, :], xs[:-1] + [1, nr_mix])
        m1 = tf.reshape(means[:, :, :, 1, :] +
                        coeffs[:, :, :, 0, :] * targets[:, :, :, 0, :],
                        xs[:-1] + [1, nr_mix])
        m2 = tf.reshape(means[:, :, :, 2, :] +
                        coeffs[:, :, :, 1, :] * targets[:, :, :, 0, :] +
                        coeffs[:, :, :, 2, :] * targets[:, :, :, 1, :],
                        xs[:-1] + [1, nr_mix])
        means = tf.concat([m0, m1, m2], 3)

        if base_distribution == 'gaussian':
            log_norm_consts = -lsigmas - 0.5 * np.log(2.0 * np.pi)
            log_kernel = -0.5 * tf.square((targets - means) / sigmas)
        elif base_distribution == 'laplace':
            log_norm_consts = -lsigmas - np.log(2.0)
            log_kernel = -tf.abs(targets - means) / sigmas
        elif base_distribution == 'logistic':
            log_norm_consts = -lsigmas
            diff = (targets - means) / sigmas
            log_kernel = -tf.nn.softplus(diff) - tf.nn.softplus(-diff)
        else:
            raise NotImplementedError
        log_exp_terms = log_kernel + log_norm_consts + logits
        log_likelihoods = tf.reduce_logsumexp(log_exp_terms, -1) - \
            tf.reduce_logsumexp(logits, -1)
    return log_likelihoods


def sample_mixture_3c(params, base_distribution='gaussian',
                      name='mixture_likelihood'):
    base_distribution = base_distribution.lower()
    ps = params.get_shape().as_list()
    xs = ps[:-1] + [3]
    nr_mix = ps[-1] // 10
    with tf.variable_scope(name):
        logits = tf.reshape(params[:, :, :, :nr_mix], xs[:-1] + [1, nr_mix])
        params = tf.reshape(params[:, :, :, nr_mix:], xs + [3 * nr_mix])
        means, lsigmas, coeffs = tf.split(params, 3, 4)
        sigmas = tf.exp(lsigmas)
        coeffs = tf.nn.tanh(coeffs)

        logits = tf.reshape(logits, [-1, nr_mix])
        sel = tf.multinomial(logits, 1)
        sel = tf.reshape(sel, xs[:-1] + [1])
        sel = tf.one_hot(sel, depth=nr_mix, axis=-1, dtype=tf.float32)

        means = tf.reduce_sum(means * sel, axis=-1)
        sigmas = tf.reduce_sum(sigmas * sel, axis=-1)
        coeffs = tf.reduce_sum(coeffs * sel, axis=-1)

        # Sample from base distribution.
        if base_distribution == 'gaussian':
            zs = tf.random_normal(xs)
        elif base_distribution == 'laplace':
            zs = tf.log(tf.random_uniform(xs)) - \
                tf.log(tf.random_uniform(xs))
        elif base_distribution == 'logistic':
            x = tf.random_uniform(xs)
            zs = tf.log(x) - tf.log(1.0 - x)
        else:
            raise NotImplementedError

        samp = sigmas * zs + means
        s0 = samp[:, :, :, 0]
        s1 = samp[:, :, :, 1] + \
            coeffs[:, :, :, 0] * s0
        s2 = samp[:, :, :, 2] + \
            coeffs[:, :, :, 1] * s0 + \
            coeffs[:, :, :, 2] * s1

        s0 = tf.reshape(s0, xs[:-1] + [1])
        s1 = tf.reshape(s1, xs[:-1] + [1])
        s2 = tf.reshape(s2, xs[:-1] + [1])

        samp = tf.concat([s0, s1, s2], 3)

        return samp


def mixture_likelihoods_1c(params, targets, base_distribution='gaussian',
                           name='mixture_likelihood'):
    """Given log-unnormalized mixture weights, shift, and log scale parameters
    for mixture components, return the likelihoods for targets.
    Args:
        params: N x H x W x 3*ncomp tensor of parameters of mixture model
        targets: N x H x W x 1 tensor of 2d targets to get likelihoods for.
        base_distribution: {'gaussian', 'laplace', 'logistic'} the base
            distribution of mixture components.
    Return:
        likelihoods: N x H x W x 1  tensor of likelihoods.
    """
    base_distribution = base_distribution.lower()
    xs = targets.get_shape().as_list()
    ps = params.get_shape().as_list()
    nr_mix = ps[-1] // 3
    with tf.variable_scope(name):
        # Compute likelihoods per target and component
        # Write log likelihood as logsumexp.
        targets = tf.reshape(targets, xs + [1])  # [B,H,W,1,1]
        logits = tf.reshape(params[:, :, :, :nr_mix], xs[:-1] + [1, nr_mix])
        params = tf.reshape(params[:, :, :, nr_mix:], xs + [2 * nr_mix])
        means, lsigmas = tf.split(params, 2, 4)  # [B,H,W,1,nr]
        sigmas = tf.exp(lsigmas)

        if base_distribution == 'gaussian':
            log_norm_consts = -lsigmas - 0.5 * np.log(2.0 * np.pi)
            log_kernel = -0.5 * tf.square((targets - means) / sigmas)
        elif base_distribution == 'laplace':
            log_norm_consts = -lsigmas - np.log(2.0)
            log_kernel = -tf.abs(targets - means) / sigmas
        elif base_distribution == 'logistic':
            log_norm_consts = -lsigmas
            diff = (targets - means) / sigmas
            log_kernel = -tf.nn.softplus(diff) - tf.nn.softplus(-diff)
        else:
            raise NotImplementedError
        log_exp_terms = log_kernel + log_norm_consts + logits
        log_likelihoods = tf.reduce_logsumexp(log_exp_terms, -1) - \
            tf.reduce_logsumexp(logits, -1)
    return log_likelihoods


def sample_mixture_1c(params, base_distribution='gaussian',
                      name='mixture_likelihood'):
    base_distribution = base_distribution.lower()
    ps = params.get_shape().as_list()
    xs = ps[:-1] + [1]
    nr_mix = ps[-1] // 3
    with tf.variable_scope(name):
        logits = tf.reshape(params[:, :, :, :nr_mix], xs[:-1] + [1, nr_mix])
        params = tf.reshape(params[:, :, :, nr_mix:], xs + [2 * nr_mix])
        means, lsigmas = tf.split(params, 2, 4)
        sigmas = tf.exp(lsigmas)

        logits = tf.reshape(logits, [-1, nr_mix])
        sel = tf.multinomial(logits, 1)
        sel = tf.reshape(sel, xs[:-1] + [1])  # [B,H,W,1,nr]
        sel = tf.one_hot(sel, depth=nr_mix, axis=-1, dtype=tf.float32)

        means = tf.reduce_sum(means * sel, axis=-1)
        sigmas = tf.reduce_sum(sigmas * sel, axis=-1)

        # Sample from base distribution.
        if base_distribution == 'gaussian':
            zs = tf.random_normal(xs)
        elif base_distribution == 'laplace':
            zs = tf.log(tf.random_uniform(xs)) - \
                tf.log(tf.random_uniform(xs))
        elif base_distribution == 'logistic':
            x = tf.random_uniform(xs)
            zs = tf.log(x) - tf.log(1.0 - x)
        else:
            raise NotImplementedError

        samp = sigmas * zs + means

        return samp


def mixture_likelihoods(params, targets, channels, base_distribution='gaussian',
                        name='mixture_likelihood'):
    if channels == 1:
        return mixture_likelihoods_1c(params, targets, base_distribution, name)
    elif channels == 3:
        return mixture_likelihoods_3c(params, targets, base_distribution, name)
    else:
        raise Exception()


def sample_mixture(params, channels, base_distribution='gaussian',
                   name='mixture_likelihood'):
    if channels == 1:
        return sample_mixture_1c(params, base_distribution, name)
    elif channels == 3:
        return sample_mixture_3c(params, base_distribution, name)
    else:
        raise Exception()
