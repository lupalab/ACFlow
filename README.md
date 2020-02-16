# ACFlow: Flow Models for Arbitrary Conditional Likelihoods

This is the official implementation of [ACFlow](https://arxiv.org/abs/1909.06319). 

## Get Started

### Prerequisites

refer to `requirements.txt`.

### Download data

download `CelebA`, `CIFAR10`, `MNIST` and `Omniglot` to your local workspace. You might need to change the path for each dataset in `datasets` folder accordingly.

MNIST and CIFAR10 can be downloaded by `torchvision`. Links for [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [Omniglot](https://github.com/renmengye/few-shot-ssl-public#omniglot) are provided here. Please cite their work if you use this repo.

## Train and Test

You can train your own model by the scripts provided below. Or you can download our pretrained weights form [here](https://drive.google.com/drive/folders/1yUqYgOT1kaBHakLShXHd-KUUM1nGVISy?usp=sharing).

### CelebA

- Train with Gaussian base likelihood

``` bash
python scripts/train.py --cfg_file=./exp/celeba/rnvp/params.json
```

- Train with autoregressive likelihood

``` bash
python scripts/train_tan.py --cfg_file=./exp/celeba/tan/params.json
```

- Compute log likelihood on testset and compute the PSNR and PRD scores using samples.

``` bash
python  scripts/test.py --cfg_file=./exp/celeba/rnvp/params.json
```

NOTE: you can run this script for multiple times with different random seed to get mean score and standard deviation.

- Compute joint likelihood p(x).

``` bash
python scripts/test_joint.py --cfg_file=./exp/celeba/rnvp/params.json
```

- Sample from arbitrary conditional distribution p(x_u | x_o) for multiple imputation.

``` bash
python scripts/sample.py --cfg_file=./exp/celeba/rnvp/params.json
```

- Sample the 'Best Guess' single imputation.

``` bash
python scripts/sample_single.py --cfg_file=./exp/celeba/rnvp/params.json
```

- Sample from joint distribution p(x).

``` bash
python scripts/sample_joint.py --cfg_file=./exp/celeba/rnvp/params.json
```

- Gibbs sampling

``` bash
python scripts/gibbs_sampling.py --cfg_file=./exp/celeba/rnvp/params.json
```

Sample the upper and lower half condition on the remaining half.

![Gibbs Sampling](imgs/gibbs.gif)

### MNIST

similar commands can be run. Config files are provided in `exp/mnist` folder.

### Omniglot

similar commands can be run. Config files are provided in `exp/omniglot` folder.

### CIFAR10

similar commands can be run. Config files are provided in `exp/cifar` folder.


## Acknowledgements

Code for evaluating [FID](https://github.com/bioinf-jku/TTUR) and [PRD](https://github.com/msmsajjadi/precision-recall-distributions) are adapted from their public implementations. Please cite their work if you use this repo.