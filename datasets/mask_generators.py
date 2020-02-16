import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import itertools

# Image inpainting mask generators
# 0: unobserved
# 1: observed


class CheckerboardGenerator:
    def __init__(self, block_size=1):
        self.block_size = block_size

    def __call__(self, image):
        H, W, C = image.shape
        b = self.block_size
        n = H // b
        assert H % b == 0
        mask = np.arange(n).reshape([n,1]) + np.arange(n)
        mask = np.mod(mask + np.random.choice([0,1]), 2)
        mask = np.ones([b,b,1,1]) * mask
        mask = np.transpose(mask, [2,0,3,1])
        mask = np.reshape(mask, [H, W, 1])
        mask = np.repeat(mask, C, axis=-1)
        return mask.astype('uint8')


class ImageMCARGenerator:
    """
    Samples mask from component-wise independent Bernoulli distribution
    with probability of _pixel_ to be unobserved p.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, image):
        gen_shape = list(image.shape)
        num_channels = gen_shape[-1]
        gen_shape[-1] = 1
        bernoulli_mask_numpy = np.random.choice(2, size=gen_shape,
                                                p=[self.p, 1 - self.p])
        mask = np.repeat(bernoulli_mask_numpy, num_channels, axis=-1)
        return mask.astype('uint8')


class FixedRectangleGenerator:
    """
    Always return an inpainting mask where unobserved region is
    a rectangle with corners in (x1, y1) and (x2, y2).
    """

    def __init__(self, y1, x1, y2, x2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, image):
        mask = np.ones_like(image)
        mask[self.y1:self.y2, self.x1:self.x2, :] = 0
        return mask.astype('uint8')


class CutoutGenerator:
    def __init__(self, cutout=16):
        self.cutout = cutout

    def __call__(self, image):
        height, width, num_channels = image.shape
        mask = np.ones_like(image)
        x = np.random.randint(width - self.cutout)
        y = np.random.randint(height - self.cutout)
        mask[y:y + self.cutout, x:x + self.cutout, :] = 0
        return mask.astype('uint8')


class RectangleGenerator:
    """
    Generates for each object a mask where unobserved region is
    a rectangle which square divided by the image square is in
    interval [min_rect_rel_square, max_rect_rel_square].
    """

    def __init__(self, min_rect_rel_square=0.3, max_rect_rel_square=1):
        self.min_rect_rel_square = min_rect_rel_square
        self.max_rect_rel_square = max_rect_rel_square

    def gen_coordinates(self, width, height):
        x1, x2 = np.random.randint(0, width, 2)
        y1, y2 = np.random.randint(0, height, 2)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        return int(x1), int(y1), int(x2), int(y2)

    def __call__(self, image):
        height, width, num_channels = image.shape
        mask = np.ones_like(image)
        x1, y1, x2, y2 = self.gen_coordinates(width, height)
        sqr = width * height
        while not (self.min_rect_rel_square * sqr <=
                   (x2 - x1 + 1) * (y2 - y1 + 1) <=
                   self.max_rect_rel_square * sqr):
            x1, y1, x2, y2 = self.gen_coordinates(width, height)
        mask[y1: y2 + 1, x1: x2 + 1, :] = 0
        return mask.astype('uint8')


class RandomPattern:
    """
    Reproduces "random pattern mask" for inpainting, which was proposed in
    Pathak, D., Krahenbuhl, P., Donahue, J., Darrell, T.,
    & Efros, A. A. Context Encoders: Feature Learning by Inpainting.
    Conference on Computer Vision and Pattern Recognition, 2016.
    ArXiv link: https://arxiv.org/abs/1604.07379

    This code is based on lines 273-283 and 316-330 of Context Encoders
    implementation:
    https://github.com/pathak22/context-encoder/blob/master/train_random.lua

    The idea is to generate small matrix with uniform random elements,
    then resize it using bicubic interpolation into a larger matrix,
    then binarize it with some threshold,
    and then crop a rectangle from random position and return it as a mask.
    If the rectangle contains too many or too few ones, the position of
    the rectangle is generated again.

    The big matrix is resampled when the total number of elements in
    the returned masks times update_freq is more than the number of elements
    in the big mask. This is done in order to find balance between generating
    the big matrix for each mask (which is involves a lot of unnecessary
    computations) and generating one big matrix at the start of the training
    process and then sampling masks from it only (which may lead to
    overfitting to the specific patterns).
    """

    def __init__(self, max_size=10000, resolution=0.06,
                 density=0.25, update_freq=1, seed=239):
        """
        Args:
            max_size (int):      the size of big binary matrix
            resolution (float):  the ratio of the small matrix size to
                                 the big one. Authors recommend to use values
                                 from 0.01 to 0.1.
            density (float):     the binarization threshold, also equals
                                 the average ones ratio in the mask
            update_freq (float): the frequency of the big matrix resampling
            seed (int):          random seed
        """
        self.max_size = max_size
        self.resolution = resolution
        self.density = density
        self.update_freq = update_freq
        self.rng = np.random.RandomState(seed)
        self.regenerate_cache()

    def regenerate_cache(self):
        """
        Resamples the big matrix and resets the counter of the total
        number of elements in the returned masks.
        """
        low_size = int(self.resolution * self.max_size)
        low_pattern = self.rng.uniform(0, 1, size=(low_size, low_size))
        low_pattern = low_pattern.astype('float32')
        pattern = Image.fromarray(low_pattern)
        pattern = pattern.resize((self.max_size, self.max_size), Image.BICUBIC)
        pattern = np.array(pattern)
        pattern = (pattern < self.density).astype('float32')
        self.pattern = pattern
        self.points_used = 0

    def __call__(self, image, density_std=0.05):
        """
        Image is supposed to have shape [H, W, C].
        Return binary mask of the same shape, where for each object
        the ratio of ones in the mask is in the open interval
        (self.density - density_std, self.density + density_std).
        The less is density_std, the longer is mask generation time.
        For very small density_std it may be even infinity, because
        there is no rectangle in the big matrix which fulfills
        the requirements.
        """
        height, width, num_channels = image.shape
        x = self.rng.randint(0, self.max_size - width + 1)
        y = self.rng.randint(0, self.max_size - height + 1)
        res = self.pattern[y:y + height, x:x + width]
        coverage = res.mean()
        while not (self.density - density_std < coverage < self.density + density_std):
            x = self.rng.randint(0, self.max_size - width + 1)
            y = self.rng.randint(0, self.max_size - height + 1)
            res = self.pattern[y:y + height, x:x + width]
            coverage = res.mean()
        mask = np.tile(res[:, :, None], [1, 1, num_channels])
        mask = 1. - mask
        self.points_used += width * height
        if self.update_freq * (self.max_size ** 2) < self.points_used:
            self.regenerate_cache()
        return mask.astype('uint8')


# Mixture mask generator

class MixtureMaskGenerator:
    """
    For each object firstly sample a generator according to their weights,
    and then sample a mask from the sampled generator.
    """

    def __init__(self, generators, weights):
        self.generators = generators
        self.weights = weights

    def __call__(self, image):
        w = np.array(self.weights, dtype='float')
        w /= w.sum()
        c_ids = np.random.choice(w.size, 1, p=w)[0]
        gen = self.generators[c_ids]
        mask = gen(image)
        return mask


# Mixtures of mask generators from different papers

class GFCGenerator:
    def __init__(self):
        gfc_o1 = FixedRectangleGenerator(26, 17, 58, 36)  # 64 x 38
        gfc_o2 = FixedRectangleGenerator(26, 29, 58, 48)  # 64 x 38
        gfc_o3 = FixedRectangleGenerator(26, 15, 37, 50)  # 22 x 70
        gfc_o4 = FixedRectangleGenerator(26, 15, 37, 34)  # 22 x 38
        gfc_o5 = FixedRectangleGenerator(26, 31, 37, 50)  # 22 x 38
        gfc_o6 = FixedRectangleGenerator(43, 20, 62, 44)  # 38 x 48

        self.generator = MixtureMaskGenerator([
            gfc_o1, gfc_o2, gfc_o3, gfc_o4, gfc_o5, gfc_o6
        ], [1] * 6)

    def __call__(self, image):
        return self.generator(image)


class SIIDGMGenerator:
    def __init__(self):
        random_pattern = RandomPattern(max_size=10000, resolution=0.06)
        mcar = ImageMCARGenerator(0.8)
        center = FixedRectangleGenerator(16, 16, 48, 48)
        half_01 = FixedRectangleGenerator(0, 0, 64, 32)
        half_02 = FixedRectangleGenerator(0, 0, 32, 64)
        half_03 = FixedRectangleGenerator(0, 32, 64, 64)
        half_04 = FixedRectangleGenerator(32, 0, 64, 64)

        self.generator = MixtureMaskGenerator([
            center, random_pattern, mcar, half_01, half_02, half_03, half_04
        ], [2, 2, 2, 1, 1, 1, 1])

    def __call__(self, image):
        return self.generator(image)


class CelebAMaskGenerator:
    def __init__(self):
        siidgm = SIIDGMGenerator()
        gfc = GFCGenerator()
        common = RectangleGenerator()
        self.generator = MixtureMaskGenerator([siidgm, gfc, common], [1, 1, 2])

    def __call__(self, image):
        return self.generator(image)

class CifarMaskGenerator:
    def __init__(self):
        mcar = ImageMCARGenerator(0.3)
        half_01 = FixedRectangleGenerator(0, 0, 32, 16)
        half_02 = FixedRectangleGenerator(0, 0, 16, 32)
        half_03 = FixedRectangleGenerator(0, 16, 32, 32)
        half_04 = FixedRectangleGenerator(16, 0, 32, 32)
        cutout = CutoutGenerator(16)
        common = RectangleGenerator(0.1, 0.5)

        self.generator = MixtureMaskGenerator(
            [mcar, half_01, half_02, half_03, half_04, cutout, common],
            [2, 1, 1, 1, 1, 2, 2])

    def __call__(self, image):
        return self.generator(image)


class MnistMaskGenerator:
    def __init__(self):
        mcar = ImageMCARGenerator(0.5)
        half_01 = FixedRectangleGenerator(0, 0, 32, 16)
        half_02 = FixedRectangleGenerator(0, 0, 16, 32)
        half_03 = FixedRectangleGenerator(0, 16, 32, 32)
        half_04 = FixedRectangleGenerator(16, 0, 32, 32)
        cutout = CutoutGenerator(16)
        common = RectangleGenerator()

        self.generator = MixtureMaskGenerator(
            [mcar, half_01, half_02, half_03, half_04, cutout, common],
            [2, 1, 1, 1, 1, 2, 2])

    def __call__(self, image):
        return self.generator(image)


class OmniglotMaskGenerator:
    def __init__(self):
        mcar = ImageMCARGenerator(0.5)
        half_01 = FixedRectangleGenerator(0, 0, 32, 16)
        half_02 = FixedRectangleGenerator(0, 0, 16, 32)
        half_03 = FixedRectangleGenerator(0, 16, 32, 32)
        half_04 = FixedRectangleGenerator(16, 0, 32, 32)
        cutout = CutoutGenerator(16)
        common = RectangleGenerator(0.1, 0.6)

        self.generator = MixtureMaskGenerator(
            [mcar, half_01, half_02, half_03, half_04, cutout, common],
            [2, 1, 1, 1, 1, 2, 2])

    def __call__(self, image):
        return self.generator(image)

