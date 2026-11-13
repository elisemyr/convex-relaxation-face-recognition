"""
Splitting people and images

The people are split into three groups:
- employees: enrolled in the gallery, they should be accepted
- validation visitors: never enrolled, used only to choose the threshold tau
- test visitors: never enrolled, used only for the final numbers

Using two different groups of visitors matters. If tau is tuned on the same
visitors that we report on, the false accept rate looks better than it is

The images of each employee are split into gallery, validation and test
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Split:
    employees: np.ndarray      # person ids of the employees
    val_visitors: np.ndarray   # person ids of the validation visitors
    test_visitors: np.ndarray  # person ids of the test visitors
    gallery_idx: np.ndarray    # image indices used to build the dictionary
    emp_val_idx: np.ndarray    # employee images used to choose tau
    emp_test_idx: np.ndarray   # employee images used for the final numbers
    vis_val_idx: np.ndarray    # visitor images used to choose tau
    vis_test_idx: np.ndarray   # visitor images used for the final numbers


def make_split(labels, n_employees, n_val_visitors, n_test_visitors,
               gallery_per_person, seed, gallery_size=None, max_test_per_person=None):
    """
    Build one random split.

    gallery_per_person is the size of the gallery pool of each employee
    max_test_per_person caps the validation and test images of each employee
    gallery_size lets us use only the first images of that pool. The test
    images stay the same for every gallery_size, so experiment E3 compares
    gallery sizes on exactly the same test faces
    """
    rng = np.random.default_rng(seed)
    people = np.unique(labels)
    needed = n_employees + n_val_visitors + n_test_visitors
    if needed > len(people):
        raise ValueError(f"Need {needed} people but the data has {len(people)}.")

    if gallery_size is None:
        gallery_size = gallery_per_person
    if gallery_size > gallery_per_person:
        raise ValueError("gallery_size cannot be larger than gallery_per_person.")

    order = rng.permutation(people)
    employees = np.sort(order[:n_employees])
    val_visitors = np.sort(order[n_employees:n_employees + n_val_visitors])
    test_visitors = np.sort(order[n_employees + n_val_visitors:needed])

    def capped(idx):
        if max_test_per_person is None:
            return idx
        return idx[:max_test_per_person]

    gallery, emp_val, emp_test = [], [], []
    for person in employees:
        idx = rng.permutation(np.flatnonzero(labels == person))
        if len(idx) < gallery_per_person + 2:
            raise ValueError(f"Person {person} has only {len(idx)} images.")
        gallery.append(idx[:gallery_size])
        rest = idx[gallery_per_person:]
        half = len(rest) // 2
        emp_val.append(capped(rest[:half]))
        emp_test.append(capped(rest[half:]))

    # Visitors keep all their images. More visitor images give a more precise
    # false accept rate, which matters when the target is as low as 1%
    vis_val = [rng.permutation(np.flatnonzero(labels == p)) for p in val_visitors]
    vis_test = [rng.permutation(np.flatnonzero(labels == p)) for p in test_visitors]

    return Split(
        employees=employees,
        val_visitors=val_visitors,
        test_visitors=test_visitors,
        gallery_idx=np.concatenate(gallery),
        emp_val_idx=np.concatenate(emp_val),
        emp_test_idx=np.concatenate(emp_test),
        vis_val_idx=np.concatenate(vis_val),
        vis_test_idx=np.concatenate(vis_test),
    )


def split_from_config(labels, cfg, seed, gallery_size=None):
    """Shortcut that reads the split sizes from the config"""
    return make_split(
        labels,
        n_employees=cfg.n_employees,
        n_val_visitors=cfg.n_val_visitors,
        n_test_visitors=cfg.n_test_visitors,
        gallery_per_person=cfg.gallery_per_person,
        seed=seed,
        gallery_size=gallery_size,
        max_test_per_person=cfg.max_test_per_person,
    )