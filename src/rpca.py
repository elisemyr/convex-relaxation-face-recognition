"""
Robust PCA, used to clean the gallery before building the dictionary

The images of one person, put as the columns of a matrix M, are close to a low-rank matrix L. Shadows and bright
spots are large but cover few pixels, so they form a sparse matrix S

    min_{L, S}  ||L||_* + lam ||S||_1   subject to  L + S = M

Course link: rank(L) is replaced by its convex envelope, the nuclear norm
||L||_* = sum of singular values, exactly as in the Matrix Completion lecture
||S||_0 is replaced by ||S||_1, as in the Compressed Sensing lectures

We solve it with the inexact augmented Lagrange multiplier method,
Lin, Chen and Ma 2010. Each step is a closed form:
- the L step is singular value thresholding, the prox of the nuclear norm
- the S step is soft thresholding, the prox of the l1 norm
"""

import numpy as np

from .solvers import soft_threshold


def singular_value_threshold(X, t):
    """Proximal operator of t ||.||_*: shrink every singular value by t"""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s = np.maximum(s - t, 0.0)
    rank = int((s > 0).sum())
    return (U[:, :rank] * s[:rank]) @ Vt[:rank], rank


def rpca(M, lam=None, tol=1e-7, max_iter=500):
    """
    Split M into a low-rank part L and a sparse part S

    lam defaults to 1 / sqrt(max(m, n)), t
    Returns L, S and a dict with the number of iterations
    and the final rank.
    """
    M = np.asarray(M, dtype=np.float64)
    m, n = M.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))

    norm_fro = np.linalg.norm(M)
    norm_two = np.linalg.norm(M, ord=2)
    # Starting dual variable, as in the paper.
    Y = M / max(norm_two, np.abs(M).max() / lam)
    mu = 1.25 / norm_two
    mu_max = mu * 1e7
    rho = 1.5

    S = np.zeros_like(M)
    rank = 0
    for it in range(1, max_iter + 1):
        L, rank = singular_value_threshold(M - S + Y / mu, 1.0 / mu)
        S = soft_threshold(M - L + Y / mu, lam / mu)
        Z = M - L - S
        Y = Y + mu * Z
        mu = min(mu * rho, mu_max)
        if np.linalg.norm(Z) / norm_fro < tol:
            break
    return L, S, {"iterations": it, "rank": rank}


def clean_gallery(X, labels, lam=None):
    """
    Run Robust PCA person by person on the gallery

    X has shape (N, d), one face per row. For each person we build the matrix
    of their faces, one face per column, and keep only the low-rank part
    Returns the cleaned gallery with the same shape as X, and the sparse parts
    """
    X = np.asarray(X, dtype=np.float64)
    X_clean = np.empty_like(X)
    X_sparse = np.empty_like(X)
    for person in np.unique(labels):
        idx = np.flatnonzero(labels == person)
        L, S, _ = rpca(X[idx].T, lam=lam)
        X_clean[idx] = L.T
        X_sparse[idx] = S.T
    return X_clean, X_sparse