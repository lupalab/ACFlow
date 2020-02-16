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

from datasets import get_dataset
from models import get_model
from utils.hparams import HParams

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_batches', type=int, default=1)
parser.add_argument('--num_samples', type=int, default=10)
parser.add_argument('--num_vis', type=int, default=256)
parser.add_argument('--sample_std', type=float, default=1.0)
args = parser.parse_args()
params = HParams(args.cfg_file)
params.batch_size = args.batch_size
params.num_samples = args.num_samples
params.sample_std = args.sample_std
pprint(params.dict)

os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu
np.random.seed(params.seed)
tf.set_random_seed(params.seed)

############################################################
save_dir = os.path.join(params.exp_dir, f'single_sample_{args.sample_std}')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

############################################################

trainset = get_dataset('train', params)
validset = get_dataset('valid', params)
testset = get_dataset('test', params)

model = get_model(params)
model.build(trainset, validset, testset)

##########################################################


def batch_show(path, sam, gt, mask, num):
    sam = sam.astype(np.uint8)
    B, H, W, C = sam.shape
    num = min(num, B)
    for n in range(num):
        cur_gt = gt[n]
        cur_sam = sam[n]
        cur_mask = mask[n]
        cur_in = cur_gt * cur_mask + 128 * (1 - cur_mask)
        img = np.concatenate([cur_in, cur_sam, cur_gt], axis=1)
        if C == 1:
            img = np.squeeze(img, axis=-1)
            plt.imsave(path + f'/{n}.png', img, cmap=plt.cm.gray)
        else:
            plt.imsave(path + f'/{n}.png', img)

    return num


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

testset.initialize(sess)
num_steps = testset.num_steps
if args.num_batches < 0:
    num_batches = num_steps
else:
    num_batches = min(num_steps, args.num_batches)

num_show = 0
for i in range(num_batches):
    sam, gt, mask = sess.run(
        [model.test_sam_mean, testset.x, testset.m])

    if num_show < args.num_vis:
        num_show += batch_show(save_dir, sam, gt, mask, args.num_vis)
