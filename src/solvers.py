"""
Solvers for the convex problems of the project

Course link: the ideal problem is min ||x||_0 subject to y = Phi x, which is
NP-hard. We replace ||x||_0 by its convex envelope on the unit ball, the l1
norm. This gives Basis Pursuit, or the Lasso when the data is noisy:

    min_x  1/2 ||Phi x - y||_2^2 + lam ||x||_1
"""

import numpy as np


def soft_threshold(z, t):
    """Proximal operator of t ||.||_1: shrink every entry towards 0 by t"""
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def lipschitz_constant(A):
    """Largest singular value of A, squared. It is the step size bound of FISTA"""
    return float(np.linalg.norm(A, ord=2) ** 2)


def fista(A, Y, lam, n_iter=300, tol=1e-6, L=None):
    """
    Solve  min_x 1/2 ||A x - y||^2 + lam ||x||_1  with FISTA

    A has shape (m, n). Y has shape (m,) for one problem, or (m, k) to solve
    k problems at once, one per column. Solving them together is much faster
    because numpy does one big matrix product per iteration.

    lam is a number, or an array of shape (n,) to give each coordinate its own
    weight. We use this in extended_l1 to weight x and e differently
    """
    single = Y.ndim == 1
    if single:
        Y = Y[:, None]
    m, n = A.shape
    lam = np.asarray(lam, dtype=np.float64)
    if lam.ndim == 1:
        lam = lam[:, None]
    if L is None:
        L = lipschitz_constant(A)

    AtY = A.T @ Y
    # When n is small, the Gram matrix A^T A makes each iteration cheaper
    gram = A.T @ A if n < 2 * m else None

    x = np.zeros((n, Y.shape[1]))
    z = x.copy()
    t = 1.0
    for _ in range(n_iter):
        grad = (gram @ z - AtY) if gram is not None else A.T @ (A @ z - Y)
        x_new = soft_threshold(z - grad / L, lam / L)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        change = np.max(np.abs(x_new - x))
        x, t = x_new, t_new
        if change < tol:
            break
    return x[:, 0] if single else x


def extended_l1(Phi, Y, lam, lam_e=None, n_iter=300, tol=1e-6):
    """
    Occlusion model:  y = Phi x + e  with x and e sparse

    We stack B = [Phi | I] and solve the Lasso on w = [x ; e]:
        min 1/2 ||B w - y||^2 + lam ||x||_1 + lam_e ||e||_1

    A mask or a scarf changes only some pixels, so e is sparse in the pixel
    basis. This only makes sense with pixel features, not PCA or random ones,
    because those mix all pixels together and e is no longer sparse

    Returns x with shape (N, k) and e with shape (m, k)
    """
    m, N = Phi.shape
    if lam_e is None:
        lam_e = lam
    B = np.hstack([Phi, np.eye(m)])
    weights = np.concatenate([np.full(N, lam), np.full(m, lam_e)])
    # The largest singular value of [Phi | I] is sqrt(sigma_max(Phi)^2 + 1).
    L = lipschitz_constant(Phi) + 1.0
    W = fista(B, Y, weights, n_iter=n_iter, tol=tol, L=L)
    return W[:N], W[N:]


def lasso_sklearn(A, y, lam, max_iter=10000):
    """
    Same problem with scikit-learn, for one y
    sklearn minimizes 1/(2m) ||y - A x||^2 + alpha ||x||_1,
    so alpha = lam / m gives the same solution as fista
    """
    from sklearn.linear_model import Lasso

    m = A.shape[0]
    model = Lasso(alpha=lam / m, fit_intercept=False, max_iter=max_iter, tol=1e-8)
    model.fit(A, y)
    return model.coef_


def lasso_cvxpy(A, y, lam):
    """Same Lasso problem with cvxpy. Slow but exact, used only in the tests"""
    import cvxpy as cp

    x = cp.Variable(A.shape[1])
    cp.Problem(cp.Minimize(0.5 * cp.sum_squares(A @ x - y) + lam * cp.norm1(x))).solve()
    return x.value


def basis_pursuit_cvxpy(A, y):
    """Basis Pursuit from the course: min ||x||_1 subject to A x = y"""
    import cvxpy as cp

    x = cp.Variable(A.shape[1])
    cp.Problem(cp.Minimize(cp.norm1(x)), [A @ x == y]).solve()
    return x.value


class RidgeSolver:
    """
    The l2 version:  min_x ||Phi x - y||^2 + lam ||x||^2.

    The solution is x = (Phi^T Phi + lam I)^-1 Phi^T y. The matrix in front of y
    does not depend on y, so we compute it once in __init__. After that,
    one test face costs a single matrix product, much faster than l1, sometimes as accurate.
    """

    def __init__(self, Phi, lam):
        N = Phi.shape[1]
        self.P = np.linalg.solve(Phi.T @ Phi + lam * np.eye(N), Phi.T)

    def solve(self, Y):
        return self.P @ Y