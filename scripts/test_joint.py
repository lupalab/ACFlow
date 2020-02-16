import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import argparse
import logging
import tensorflow as tf
import numpy as np
import pickle
import gzip
from pprint import pformat, pprint
import matplotlib.pyplot as plt
from easydict import EasyDict as edict

from datasets import get_dataset
from models import get_model
from utils.hparams import HParams

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--batch_size', type=int, default=100)
args = parser.parse_args()
params = HParams(args.cfg_file)
params.seed = args.seed
params.gpu = args.gpu
params.batch_size = args.batch_size
pprint(params.dict)

os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu
np.random.seed(params.seed)
tf.set_random_seed(params.seed)

############################################################
save_dir = os.path.join(params.exp_dir, 'test_joint')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

f = open(save_dir + '/test.txt', 'w')
f.write(pformat(args))
f.write('\n')
############################################################

trainset = get_dataset('train', params)
validset = get_dataset('valid', params)
testset = get_dataset('test', params)

x_ph = tf.placeholder(tf.uint8, trainset.x.get_shape().as_list())
m_ph = tf.placeholder(tf.uint8, trainset.m.get_shape().as_list())

dummyset = edict()
dummyset.x, dummyset.m = x_ph, m_ph

model = get_model(params)
model.build(dummyset, dummyset, dummyset)

##########################################################

saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
config = tf.ConfigProto()
config.log_device_placement = True
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

weights_dir = os.path.join(params.exp_dir, 'weights')
logging.info(f'Restoring parameters from {weights_dir}')
restore_from = tf.train.latest_checkpoint(weights_dir)
saver.restore(sess, restore_from)

log_likel = []
testset.initialize(sess)
num_steps = testset.num_steps
for i in range(num_steps):
    x_nda = sess.run(testset.x)
    m_nda = np.zeros_like(x_nda)
    ll = sess.run(model.test_ll, feed_dict={x_ph:x_nda, m_ph:m_nda})
    log_likel.append(ll)
   
log_likel = np.concatenate(log_likel, axis=0)

f.write(f'log_likel: {log_likel.shape} avg: {np.mean(log_likel)}\n')
bpd = -np.mean(log_likel)/(np.prod(params.image_shape) * np.log(2.))
f.write(f'bpd: {bpd}\n')
f.close()
