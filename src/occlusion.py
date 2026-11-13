"""
Fake occlusions for the test faces

Yale B has no masks or glasses, so we draw them ourselves
Occlusions are only applied to test images, never to the gallery
In every function, level is the share of the face that gets hidden,
between 0 and 1
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def _texture(shape, rng, sigma=2.0):
    """A random smooth pattern with values in [0, 1]. Harder to spot than a flat color"""
    t = gaussian_filter(rng.random(shape), sigma=sigma)
    t -= t.min()
    return t / max(t.max(), 1e-8)


def block_occlusion(img, level, rng, fill="texture"):
    """A square at a random place that covers `level` of the image area"""
    out = img.copy()
    if level <= 0:
        return out
    h, w = img.shape
    side = int(round(np.sqrt(level * h * w)))
    side_h, side_w = min(side, h), min(side, w)
    top = rng.integers(0, h - side_h + 1)
    left = rng.integers(0, w - side_w + 1)
    if fill == "texture":
        patch = _texture((side_h, side_w), rng)
    elif fill == "black":
        patch = np.zeros((side_h, side_w))
    else:
        patch = np.ones((side_h, side_w))
    out[top:top + side_h, left:left + side_w] = patch
    return out


def mask_occlusion(img, level, rng):
    """
    A medical mask: the lower part of the face is covered
    level is the share of the image height covered, starting from the bottom.
    A real mask covers about 0.4 to 0.5 of a cropped face
    """
    out = img.copy()
    if level <= 0:
        return out
    h, _ = img.shape
    rows = int(round(level * h))
    mask = 0.75 + 0.15 * _texture((rows, img.shape[1]), rng, sigma=3)
    out[h - rows:, :] = mask
    return out


def glasses_occlusion(img, level, rng):
    """
    Dark glasses: a dark band around the eyes
    level is the band height as a share of the image height
    """
    out = img.copy()
    if level <= 0:
        return out
    h, _ = img.shape
    rows = max(1, int(round(level * h)))
    center = int(0.35 * h)
    top = max(0, center - rows // 2)
    out[top:top + rows, :] = 0.05 * _texture((min(rows, h - top), img.shape[1]), rng)
    return out


def pixel_corruption(img, level, rng):
    """A random share `level` of the pixels is replaced by random values."""
    out = img.copy()
    if level <= 0:
        return out
    hit = rng.random(img.shape) < level
    out[hit] = rng.random(hit.sum())
    return out


OCCLUSIONS = {
    "block": block_occlusion,
    "mask": mask_occlusion,
    "glasses": glasses_occlusion,
    "pixels": pixel_corruption,
}


def apply_occlusion(images, kind, level, seed=0):
    """Apply one kind of occlusion to every image of a batch. Returns a copy"""
    if kind not in OCCLUSIONS:
        raise ValueError(f"Unknown occlusion {kind}. Choose from {list(OCCLUSIONS)}.")
    rng = np.random.default_rng(seed)
    func = OCCLUSIONS[kind]
    return np.stack([func(img, level, rng) for img in images]).astype(np.float32)