"""
A pipeline is: feature extractor + optional Robust PCA cleaning + classifier

It takes images as input, so the experiments and the notebook can treat every
method the same way. build_method gives the methods used in the report
"""

import time

import numpy as np

from .baselines import DeepEmbeddingNN, EigenfacesNN, EigenfacesSVM
from .features import PixelFeatures
from .rpca import clean_gallery
from .src_classifier import CRCClassifier, SRCClassifier


class Pipeline:
    def __init__(self, classifier, features=None, use_rpca=False):
        """features=None means the classifier gets the raw images, as for FaceNet"""
        self.classifier = classifier
        self.features = features
        self.use_rpca = use_rpca

    def _transform(self, images):
        return images if self.features is None else self.features.transform(images)

    def fit(self, images, labels):
        if self.features is not None:
            self.features.fit(images)
        F = self._transform(images)
        if self.use_rpca:
            F, _ = clean_gallery(F, labels)
        self.classifier.fit(F, labels)
        return self

    def run(self, images):
        """Returns pred, score, occluded_fraction and the time per test face in seconds"""
        start = time.perf_counter()
        F = self._transform(images)
        if hasattr(self.classifier, "explain"):
            out = self.classifier.explain(F)
            result = {k: out[k] for k in ("pred", "score", "occluded_fraction")}
        else:
            pred, score = self.classifier.predict_with_score(F)
            result = {"pred": pred, "score": score, "occluded_fraction": np.zeros(len(pred))}
        result["time_per_face"] = (time.perf_counter() - start) / max(len(images), 1)
        return result


# Names used in the CSV files and in the figures.
METHODS = ["SRC", "SRC+occlusion", "SRC+occlusion+RPCA", "CRC",
           "Eigenfaces+SVM", "Eigenfaces+NN", "FaceNet+NN"]


def build_method(name, cfg, seed=0):
    """Create a fresh, unfitted pipeline for one method name"""
    pixels = lambda: PixelFeatures(cfg.work_size)
    if name == "SRC":
        return Pipeline(SRCClassifier(lam=cfg.lam_src, n_iter=cfg.fista_iter), pixels())
    if name == "SRC+occlusion":
        return Pipeline(SRCClassifier(lam=cfg.lam_src, occlusion=True, n_iter=cfg.fista_iter,
                                      e_threshold=cfg.e_pixel_threshold), pixels())
    if name == "SRC+occlusion+RPCA":
        return Pipeline(SRCClassifier(lam=cfg.lam_src, occlusion=True, n_iter=cfg.fista_iter,
                                      e_threshold=cfg.e_pixel_threshold), pixels(), use_rpca=True)
    if name == "SRC+RPCA":
        return Pipeline(SRCClassifier(lam=cfg.lam_src, n_iter=cfg.fista_iter), pixels(), use_rpca=True)
    if name == "CRC":
        return Pipeline(CRCClassifier(lam=cfg.lam_crc), pixels())
    if name == "CRC+RPCA":
        return Pipeline(CRCClassifier(lam=cfg.lam_crc), pixels(), use_rpca=True)
    if name == "Eigenfaces+SVM":
        return Pipeline(EigenfacesSVM(cfg.pca_components, seed=seed), pixels())
    if name == "Eigenfaces+NN":
        return Pipeline(EigenfacesNN(cfg.pca_components, seed=seed), pixels())
    if name == "FaceNet+NN":
        return Pipeline(DeepEmbeddingNN(), features=None)
    raise ValueError(f"Unknown method {name}.")