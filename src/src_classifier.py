"""
Sparse Representation-based Classification and the access decision

How SRC decides who a face belongs to, for a test face y:
1. write y as a combination of all gallery faces: y = Phi x + noise
2. find a sparse x with the l1 solver
3. for each person c, rebuild y with only the coefficients of c,
   and measure the residual r_c = ||y - Phi delta_c(x)||
4. the person with the smallest residual is the best match

How we decide if the match can be trusted:
the Sparsity Concentration Index, SCI
    SCI(x) = (C * max_c ||delta_c(x)||_1 / ||x||_1 - 1) / (C - 1)
SCI = 1 when all the weight is on one person, so a confident match
SCI = 0 when the weight is spread over everyone, so probably a stranger

All classifiers here share the same interface:
    fit(X_gallery, labels)
    explain(Y) -> dict with pred, score, residuals, coef, e, occluded_fraction
    predict_with_score(Y) -> pred, score
"""

import numpy as np

from .features import normalize_rows
from .solvers import RidgeSolver, extended_l1, fista, lipschitz_constant


def sparsity_concentration_index(coef, class_masks):
    """
    SCI for each test face
    coef has shape (N, k): one column of coefficients per test face
    class_masks is a boolean array of shape (C, N)
    """
    C = class_masks.shape[0]
    total = np.abs(coef).sum(axis=0) + 1e-12
    per_class = np.stack([np.abs(coef[mask]).sum(axis=0) for mask in class_masks])
    return (C * per_class.max(axis=0) / total - 1.0) / (C - 1.0)


def residual_ratio_score(residuals):
    """
    Another confidence score: 1 - r_best / r_second
    Close to 1 when one person explains the face much better than the others
    residuals has shape (k, C)
    """
    r = np.sort(residuals, axis=1)
    return 1.0 - r[:, 0] / np.maximum(r[:, 1], 1e-12)


class _RepresentationClassifier:
    """Common code for SRC and CRC. Children only define _solve"""

    def __init__(self, score="sci"):
        if score not in ("sci", "residual_ratio"):
            raise ValueError("score must be 'sci' or 'residual_ratio'.")
        self.score = score

    def fit(self, X, labels):
        """X has shape (N, d), one gallery face per row. labels has shape (N,)"""
        self.Phi_ = normalize_rows(np.asarray(X, dtype=np.float64)).T  # shape (d, N)
        self.labels_ = np.asarray(labels)
        self.classes_ = np.unique(self.labels_)
        self.class_masks_ = np.stack([self.labels_ == c for c in self.classes_])
        self._prepare()
        return self

    def _prepare(self):
        pass

    def _solve(self, Y):
        """Y has shape (d, k). Returns coef (N, k) and e (d, k)"""
        raise NotImplementedError

    def explain(self, X_test):
        """Run the full decision on test faces. X_test has shape (k, d)"""
        Y = normalize_rows(np.asarray(X_test, dtype=np.float64)).T
        coef, e = self._solve(Y)
        Y_clean = Y - e

        residuals = np.stack([
            np.linalg.norm(Y_clean - self.Phi_[:, mask] @ coef[mask], axis=0)
            for mask in self.class_masks_
        ], axis=1)  # shape (k, C)
        pred = self.classes_[np.argmin(residuals, axis=1)]

        if self.score == "sci":
            score = sparsity_concentration_index(coef, self.class_masks_)
        else:
            score = residual_ratio_score(residuals)

        return {
            "pred": pred,
            "score": score,
            "residuals": residuals,
            "coef": coef,
            "e": e,
            "occluded_fraction": self._occluded_fraction(e),
        }

    def _occluded_fraction(self, e):
        """Share of pixels where |e| is large. Only meaningful with the occlusion model"""
        return np.zeros(e.shape[1])

    def predict_with_score(self, X_test):
        out = self.explain(X_test)
        return out["pred"], out["score"]

    def predict(self, X_test):
        return self.explain(X_test)["pred"]


class SRCClassifier(_RepresentationClassifier):
    """
    SRC with the l1 penalty

    occlusion=False: plain SRC, min 1/2 ||Phi x - y||^2 + lam ||x||_1
    occlusion=True: extended SRC, y = Phi x + e with a sparse error e

    e_threshold: a pixel counts as occluded when |e_i| is above
    e_threshold times the average pixel size of y, which is 1 / sqrt(d)
    because y has unit norm
    """

    def __init__(self, lam=0.005, occlusion=False, lam_e=None, n_iter=300,
                 score="sci", e_threshold=0.1):
        super().__init__(score=score)
        self.lam = lam
        self.occlusion = occlusion
        self.lam_e = lam_e
        self.n_iter = n_iter
        self.e_threshold = e_threshold

    def _prepare(self):
        self.L_ = lipschitz_constant(self.Phi_)

    def _solve(self, Y):
        if self.occlusion:
            return extended_l1(self.Phi_, Y, self.lam, lam_e=self.lam_e, n_iter=self.n_iter)
        coef = fista(self.Phi_, Y, self.lam, n_iter=self.n_iter, L=self.L_)
        return coef, np.zeros_like(Y)

    def _occluded_fraction(self, e):
        d = e.shape[0]
        return (np.abs(e) > self.e_threshold / np.sqrt(d)).mean(axis=0)


class CRCClassifier(_RepresentationClassifier):
    """
    Collaborative representation, Zhang et al. 2011: same idea as SRC
    but with ridge instead of the Lasso. No sparsity, much faster
    It lets us test whether the l1 relaxation really helps
    """

    def __init__(self, lam=0.01, score="residual_ratio"):
        super().__init__(score=score)
        self.lam = lam

    def _prepare(self):
        self.solver_ = RidgeSolver(self.Phi_, self.lam)

    def _solve(self, Y):
        return self.solver_.solve(Y), np.zeros_like(Y)


def access_decision(pred, score, tau, occluded_fraction=None, rho=None, access_list=None):
    """
    Turn identities and scores into door decisions

    Rules, checked in this order:
    1. "badge" if too much of the face is hidden: occluded_fraction > rho
    2. "accept" if score >= tau and the person is allowed through this door
    3. "reject" in every other case
    """
    pred = np.asarray(pred)
    score = np.asarray(score)
    decision = np.full(len(pred), "reject", dtype=object)

    allowed = np.ones(len(pred), dtype=bool)
    if access_list is not None:
        allowed = np.isin(pred, access_list)
    decision[(score >= tau) & allowed] = "accept"

    if occluded_fraction is not None and rho is not None:
        decision[np.asarray(occluded_fraction) > rho] = "badge"
    return decision