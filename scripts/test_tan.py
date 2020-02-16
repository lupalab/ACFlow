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
from metrics.PRD import inception
from metrics.PRD import prd_score as prd

parser = argparse.ArgumentParser()
parser.add_argument('--trail', type=int, default=0)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--num_batches', type=int, default=-1)
parser.add_argument('--num_samples', type=int, default=10)
parser.add_argument('--sample_std', type=float, default=1.0)
parser.add_argument('--num_vis', type=int, default=10)
parser.add_argument('--num_clusters', type=int, default=20,
                    help='number of cluster centers to fit')
parser.add_argument('--num_angles', type=int, default=1001,
                    help='number of angles for which to compute PRD, must be '
                         'in [3, 1e6]')
parser.add_argument('--num_runs', type=int, default=10,
                    help='number of independent runs over which to average the '
                         'PRD data')
parser.add_argument('--inception_path', type=str,
                    default=f'{p}/metrics/PRD/inception.pb',
                    help='path to pre-trained Inception.pb file')
args = parser.parse_args()
params = HParams(args.cfg_file)
params.seed = args.seed
params.gpu = args.gpu
params.batch_size = args.batch_size
params.num_samples = args.num_samples
params.sample_std = args.sample_std
pprint(params.dict)

os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu
np.random.seed(params.seed)
tf.set_random_seed(params.seed)

############################################################
save_dir = os.path.join(
    params.exp_dir, f'test_{args.sample_std}_trail_{args.trail}')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

f = open(save_dir + '/test.txt', 'w')
f.write(pformat(args))
f.write('\n')
############################################################

trainset = get_dataset('train', params)
validset = get_dataset('valid', params)
testset = get_dataset('test', params)

model = get_model(params)
model.build(trainset, validset, testset)

##########################################################


def generate_inception_embedding(imgs, inception_path, layer_name='pool_3:0'):
    return inception.embed_images_in_inception(imgs, inception_path, layer_name)


def batch_psnr(sam, gt):
    '''
    sam: [B,N,H,W,C]
    gt : [B,H,W,C]
    '''
    sam = sam.astype('float32') / 255.
    gt = gt.astype('float32') / 255.
    gt = np.expand_dims(gt, axis=1)
    H, W, C = sam.shape[2:]
    err = np.sum((sam - gt)**2, axis=(2, 3, 4))
    err /= (H * W * C)
    psnr = 10 * np.log10((1.0 ** 2) / err)

    return psnr


def batch_show(path, sam, gt, mask, num):
    sam = sam.astype(np.uint8)
    B, N, H, W, C = sam.shape
    num = min(num, B)
    for n in range(num):
        cur_gt = gt[n]
        cur_sam = sam[n]
        cur_mask = mask[n]
        cur_in = cur_gt * cur_mask + 128 * (1 - cur_mask)
        cur_sam = cur_sam.transpose(1, 0, 2, 3).reshape([H, N * W, C])
        img = np.concatenate([cur_in, cur_sam, cur_gt], axis=1)
        if C == 1:
            img = np.squeeze(img, axis=-1)
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

log_likel = []
psnr = []
fake = []
real = []
testset.initialize(sess)
num_steps = testset.num_steps
if args.num_batches < 0:
    num_batches = num_steps
else:
    num_batches = min(num_steps, args.num_batches)
num_show = 0
for i in range(num_steps):
    ll, gt, mask = sess.run(
        [model.test_ll, testset.x, testset.m])
    # sam = model.sample(sess, gt, mask)
    log_likel.append(ll)
    # psnr.append(batch_psnr(sam, gt))
    # fake.append(sam)
    # real.append(gt)

    # if num_show < args.num_vis:
    #     num_show += batch_show(save_dir, sam, gt, mask, args.num_vis)

log_likel = np.concatenate(log_likel, axis=0)
# psnr = np.concatenate(psnr, axis=0)
# ind = set(range(psnr.shape[0]))
# ind = ind - set(np.where(np.isinf(psnr))[0])
# psnr = psnr[list(ind)]

f.write(f'log_likel: {log_likel.shape} avg: {np.mean(log_likel)}\n')
# f.write(f'psnr: {psnr.shape} avg: {np.mean(psnr)}\n')

# fake = np.concatenate(fake, axis=0)
# fake = fake.astype('float32')
# real = np.concatenate(real, axis=0)
# real = real.astype('float32')
# # embedding
# real_emb = generate_inception_embedding(real, args.inception_path)
# del real
# B, N, H, W, C = fake.shape
# fake = np.reshape(fake, [B * N, H, W, C])
# fake_emb = generate_inception_embedding(fake, args.inception_path)
# fake_emb = np.reshape(fake_emb, [B, N] + list(fake_emb.shape[1:]))
# del fake
# # PRD
# prd_data = []
# for i in range(args.num_samples):
#     prd_data.append(prd.compute_prd_from_embedding(
#         eval_data=fake_emb[:, i],
#         ref_data=real_emb,
#         num_clusters=args.num_clusters,
#         num_angles=args.num_angles,
#         num_runs=args.num_runs))
# prd_data = np.array(prd_data)
# # print(prd_data.shape)
# avg_prd = np.mean(prd_data, axis=0)
# f_beta = prd.prd_to_max_f_beta_pair(avg_prd[0], avg_prd[1], beta=8)
# f.write(f'F_8:{f_beta[0]} F_1/8:{f_beta[1]}\n')
f.close()