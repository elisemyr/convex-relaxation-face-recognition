"""
Loading the faces
"""

import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# A is the azimuth and E is the elevation of the light source, in degrees
_NAME_RE = re.compile(r"yaleB(\d+)_P00A([+-]\d+)E([+-]\d+)\.(pgm|png)$", re.IGNORECASE)


def load_yaleb(data_dir, size=(96, 84)):
    """
    Load the cropped Extended Yale B faces.

    Returns images, labels and a dict with the light angles of each image
    The "Ambient" images are skipped because they are not lit by a single light
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"{data_dir} not found. Download the cropped Extended Yale B "
            "and unzip it there. See README.md"
        )

    images, labels, azimuth, elevation = [], [], [], []
    seen = set()
    h, w = size
    for path in sorted(data_dir.rglob("yaleB*")):
        match = _NAME_RE.search(path.name)
        if match is None:
            # Ambient images, folders and broken files do not match the pattern
            continue
        person, az, el, _ = match.groups()
        # Skip the same image if both the .pgm and the .png copies are present
        key = (int(person), int(az), int(el))
        if key in seen:
            continue
        seen.add(key)
        try:
            img = Image.open(path).convert("L").resize((w, h), Image.BILINEAR)
        except OSError:
            # A few files in some copies of the dataset are corrupted
            continue
        images.append(np.asarray(img, dtype=np.float32) / 255.0)
        labels.append(int(person))
        azimuth.append(int(az))
        elevation.append(int(el))

    if not images:
        raise RuntimeError(f"No Yale B images found in {data_dir} or its subfolders.")

    _, labels = np.unique(np.array(labels), return_inverse=True)
    info = {"azimuth": np.array(azimuth), "elevation": np.array(elevation)}
    return np.stack(images), labels, info


def make_synthetic_faces(n_people=38, n_per_person=64, size=(96, 84), noise=0.01, seed=0):
    """
    Build fake faces that follow the same model as the course.

    Each person has an albedo map, which is a smooth random image
    Each image is this albedo times a smooth lighting pattern
    The lighting is a random mix of 9 fixed patterns, so the images of one
    person lie close to a 9-dimensional subspace, as in Basri and Jacobs 2003

    This data is only for testing the code. It says nothing about real faces
    """
    rng = np.random.default_rng(seed)
    h, w = size
    yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")

    # 9 smooth lighting patterns: constant, linear and quadratic terms
    patterns = np.stack([
        np.ones_like(xx), xx, yy, xx * yy, xx**2, yy**2,
        xx**2 - yy**2, xx * yy**2, yy * xx**2,
    ])

    # A shared "mean face": an oval that is brighter in the middle
    mean_face = np.exp(-(xx**2 / 0.5 + yy**2 / 0.8))

    images, labels = [], []
    for person in range(n_people):
        texture = gaussian_filter(rng.standard_normal((h, w)), sigma=4)
        texture /= np.abs(texture).max()
        albedo = np.clip(mean_face + 0.35 * texture, 0.05, None)
        for _ in range(n_per_person):
            weights = rng.normal(0.0, 0.35, size=9)
            weights[0] = 1.0
            light = np.clip(np.tensordot(weights, patterns, axes=1), 0.0, None)
            img = albedo * light + noise * rng.standard_normal((h, w))
            images.append(np.clip(img / max(img.max(), 1e-8), 0.0, 1.0))
            labels.append(person)

    return np.stack(images).astype(np.float32), np.array(labels), {}


def load_dataset(cfg, synthetic=False):
    """Load Yale B, or the synthetic faces when synthetic is True"""
    if synthetic:
        return make_synthetic_faces(size=cfg.load_size)
    return load_yaleb(cfg.data_dir, size=cfg.load_size)