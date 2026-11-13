"""
Turning images into feature vectors.

Three ways to get d numbers from a face, as in Wright et al. 2009:
- downsampling: keep a small version of the image
- PCA: project on the first d eigenfaces of the gallery
- random projection: multiply by a random Gaussian matrix

The random projection is the link with the RIP lecture: a Gaussian matrix
satisfies RIP with high probability when d is of order s log(N / s)

Every extractor has fit(images) and transform(images), like scikit-learn
Feature arrays have shape (n_images, d): one row per image
"""

import numpy as np
from PIL import Image

# Image sizes used by Wright et al. for each feature dimension.
DIM_TO_SIZE = {30: (6, 5), 56: (8, 7), 120: (12, 10), 504: (24, 21), 2016: (48, 42)}


def resize_images(images, size):
    """Resize a batch of images to size = (height, width)"""
    h, w = size
    if images.shape[1:] == (h, w):
        return images.astype(np.float32)
    out = [np.asarray(Image.fromarray(img.astype(np.float32), mode="F").resize((w, h), Image.BILINEAR))
           for img in images]
    return np.stack(out).astype(np.float32)


def normalize_rows(X, eps=1e-12):
    """Give every row unit l2 norm. SRC behaves badly without this"""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def size_for_dim(d, aspect=8 / 7):
    """Image size with about d pixels and the aspect ratio of Yale B crops"""
    if d in DIM_TO_SIZE:
        return DIM_TO_SIZE[d]
    h = max(1, int(round(np.sqrt(d * aspect))))
    return h, max(1, int(round(d / h)))


class PixelFeatures:
    """Downsample the image and flatten it. Keeps the pixel layout, so occlusions stay local"""

    def __init__(self, size):
        self.size = tuple(size)

    def fit(self, images):
        return self

    def transform(self, images):
        small = resize_images(images, self.size)
        return small.reshape(len(small), -1)


class PCAFeatures:
    """Project on the first d eigenfaces, computed on the gallery only"""

    def __init__(self, d):
        self.d = d

    def fit(self, images):
        X = images.reshape(len(images), -1).astype(np.float64)
        self.mean_ = X.mean(axis=0)
        # The right singular vectors of the centered data are the eigenfaces.
        _, _, vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        self.components_ = vt[: self.d]
        return self

    def transform(self, images):
        X = images.reshape(len(images), -1).astype(np.float64)
        return ((X - self.mean_) @ self.components_.T).astype(np.float32)


class RandomProjectionFeatures:
    """Multiply the flattened image by a fixed random Gaussian matrix R of shape (d, m)"""

    def __init__(self, d, seed=0):
        self.d = d
        self.seed = seed

    def fit(self, images):
        m = images[0].size
        rng = np.random.default_rng(self.seed)
        # Entries from N(0, 1/d), so that E ||R x||^2 = ||x||^2.
        self.R_ = rng.normal(0.0, 1.0 / np.sqrt(self.d), size=(self.d, m)).astype(np.float32)
        return self

    def transform(self, images):
        X = images.reshape(len(images), -1).astype(np.float32)
        return X @ self.R_.T


def make_features(kind, d, seed=0):
    """Factory used by the experiments: kind is 'pixels', 'pca' or 'random'"""
    if kind == "pixels":
        return PixelFeatures(size_for_dim(d))
    if kind == "pca":
        return PCAFeatures(d)
    if kind == "random":
        return RandomProjectionFeatures(d, seed=seed)
    raise ValueError(f"Unknown feature kind {kind}.")