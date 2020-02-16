
def get_dataset(split, hps):
    if hps.dataset == 'mnist':
        from .mnist import Dataset
        dataset = Dataset(split, hps.batch_size)
    elif hps.dataset == 'omniglot':
        from .omniglot import Dataset
        dataset = Dataset(split, hps.batch_size)
    elif hps.dataset == 'celeba':
        from .celeba import Dataset
        dataset = Dataset(split, hps.batch_size)
    elif hps.dataset == 'cifar10':
        from .cifar10 import Dataset
        dataset = Dataset(split, hps.batch_size)
    else:
        raise Exception()

    assert dataset.image_shape == hps.image_shape

    return dataset
