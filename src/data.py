"""
Loading the faces
"""

import re
from pathlib import Path

import numpy as np
from PIL import Image

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


def load_dataset(cfg):
    """Load Yale B with the sizes from the config"""
    return load_yaleb(cfg.data_dir, size=cfg.load_size)