"""
All the settings of the project live here

Every script reads this file, so one change here changes the whole pipeline
Keeping the seeds here is what makes the results reproducible
"""

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    # ---------------- data ----------------
    #the loader looks for Yale B images in every subfolder of this folder
    data_dir: Path = ROOT / "data"
    #we load at half size to save memory
    load_size: tuple = (96, 84)
    # Pixel size used by SRC and Robust PCA. 24 x 21 gives 504 pixels,
    # one of the sizes used by Wright et al. 2009. Try (48, 42) for final runs
    # if your computer is fast enough.
    work_size: tuple = (24, 21)

    # ---------------- splits ----------------
    n_employees: int = 30
    n_val_visitors: int = 4
    n_test_visitors: int = 4
    gallery_per_person: int = 32
    # Seeds for the repeated random splits, Results are averaged over them
    seeds: list = field(default_factory=lambda: [0, 1, 2, 3, 4])

    # ---------------- models ----------------
    lam_src: float = 0.005        # weight of the l1 penalty in SRC
    lam_crc: float = 0.01         # weight of the l2 penalty in ridge/CRC
    fista_iter: int = 300         # number of FISTA iterations
    pca_components: int = 120     #for eigenfaces
    # A pixel counts as occluded when |e| is above this share of the average
    # pixel size of the test face
    e_pixel_threshold: float = 0.1
    #fall back to the badge when more than this share of pixels is occluded
    rho_fallback: float = 0.4
    #target false accept rate used to choose the threshold tau
    target_far: float = 0.01

    # ---------------- experiments ----------------
    feature_dims: list = field(default_factory=lambda: [30, 56, 120, 504])
    occlusion_levels: list = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    occlusion_kinds: list = field(default_factory=lambda: ["block", "mask", "glasses", "pixels"])
    gallery_sizes: list = field(default_factory=lambda: [5, 10, 20, 32])

    max_test_per_person: int = 15

    results_dir: Path = ROOT / "results"


def quick_config() -> Config:
    """A smaller setting to check that the code runs in a few minutes because it can be slow with the full settings"""
    cfg = Config()
    cfg.seeds = [0]
    cfg.fista_iter = 150
    cfg.max_test_per_person = 5
    cfg.feature_dims = [30, 120]
    cfg.occlusion_levels = [0.0, 0.3]
    cfg.occlusion_kinds = ["block", "mask"]
    cfg.gallery_sizes = [10, 32]
    return cfg