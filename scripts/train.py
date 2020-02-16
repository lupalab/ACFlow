import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import argparse
import logging
import tensorflow as tf
import numpy as np
from pprint import pformat, pprint

from datasets import get_dataset
from models import get_model
from utils.hparams import HParams

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
args = parser.parse_args()
params = HParams(args.cfg_file)
pprint(params.dict)

os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu
np.random.seed(params.seed)
tf.set_random_seed(params.seed)

############################################################
logging.basicConfig(filename=params.exp_dir + '/train.log',
                    filemode='w',
                    level=logging.INFO,
                    format='%(message)s')
logging.info(pformat(params.dict))
writer = tf.summary.FileWriter(params.exp_dir + '/summaries')
############################################################

trainset = get_dataset('train', params)
validset = get_dataset('valid', params)
testset = get_dataset('test', params)
logging.info(f"trainset: {trainset.size} \
               validset: {validset.size} \
               testset: {testset.size}")

model = get_model(params)
model.build(trainset, validset, testset)

total_params = 0
trainable_variables = tf.trainable_variables()
logging.info('=' * 20)
logging.info("Variables:")
logging.info(pformat(trainable_variables))
for k, v in enumerate(trainable_variables):
    num_params = np.prod(v.get_shape().as_list())
    total_params += num_params

logging.info("TOTAL TENSORS: %d TOTAL PARAMS: %f[M]" % (
    k + 1, total_params / 1e6))
logging.info('=' * 20)

##########################################################


def train_epoch(sess):
    train_ll = []
    num_steps = trainset.num_steps
    trainset.initialize(sess)
    for i in range(num_steps):
        if (params.summ_freq > 0) and (i % params.summ_freq == 0):
            ll, summ, step, _ = sess.run(
                [model.train_ll, model.summ_op,
                 model.global_step, model.train_op])
            writer.add_summary(summ, step)
        else:
            ll, _ = sess.run([model.train_ll, model.train_op])
        train_ll.append(ll)
        if (params.print_freq > 0) and (i % params.print_freq == 0):
            nll = np.mean(-ll)
            logging.info('step: %d nll:%.4f ' % (i, nll))
    train_ll = np.concatenate(train_ll, axis=0)

    return np.mean(-train_ll)


def valid_epoch(sess):
    valid_ll = []
    num_steps = validset.num_steps
    validset.initialize(sess)
    for i in range(num_steps):
        ll = sess.run(model.valid_ll)
        valid_ll.append(ll)
    valid_ll = np.concatenate(valid_ll, axis=0)

    return np.mean(-valid_ll)


def test_epoch(sess):
    test_ll = []
    num_steps = testset.num_steps
    testset.initialize(sess)
    for i in range(num_steps):
        ll = sess.run(model.test_ll)
        test_ll.append(ll)
    test_ll = np.concatenate(test_ll, axis=0)

    return np.mean(-test_ll)


##########################################################
initializer = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
begin_at_epoch = 0

config = tf.ConfigProto()
config.log_device_placement = True
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sess.run(initializer)
if params.restore_from != '':
    logging.info('Restoring parameters from %s' % params.restore_from)
    if os.path.isdir(params.restore_from):
        restore_from = tf.train.latest_checkpoint(params.restore_from)
        begin_at_epoch = int(restore_from.split('-')[-1])
        saver.restore(sess, restore_from)

logging.info('starting training')
best_valid_metric = np.inf
best_test_metric = np.inf
for epoch in range(begin_at_epoch, begin_at_epoch + params.epochs):
    train_metric = train_epoch(sess)
    valid_metric = valid_epoch(sess)
    test_metric = test_epoch(sess)
    # save
    if valid_metric < best_valid_metric:
        best_valid_metric = valid_metric
        save_path = os.path.join(params.exp_dir, 'weights', 'epoch')
        saver.save(sess, save_path, global_step=epoch + 1)
    if test_metric < best_test_metric:
        best_test_metric = test_metric

    logging.info("Epoch %d, train: %.4f, valid: %.4f/%.4f, test: %.4f/%.4f" %
                 (epoch, train_metric, valid_metric, best_valid_metric,
                  test_metric, best_test_metric))
    sys.stdout.flush()
