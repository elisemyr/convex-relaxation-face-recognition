"""
Checks that the building blocks are correct. Run with:  python3 -m pytest -q

These tests use small random problems where we know the right answer.
They are also useful for the oral: they show that the solvers do what the
theory says
"""

import numpy as np

from src.evaluation import choose_tau, far_frr
from src.rpca import rpca
from src.solvers import basis_pursuit_cvxpy, extended_l1, fista, lasso_cvxpy, lasso_sklearn
from src.src_classifier import SRCClassifier, access_decision, sparsity_concentration_index


def _random_problem(m=60, n=200, s=5, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n)) / np.sqrt(m)
    x = np.zeros(n)
    x[rng.choice(n, s, replace=False)] = rng.standard_normal(s)
    return A, x, A @ x


def test_fista_matches_cvxpy_and_sklearn():
    A, _, y = _random_problem()
    y = y + 0.01 * np.random.default_rng(1).standard_normal(len(y))
    lam = 0.02
    x_f = fista(A, y, lam, n_iter=5000, tol=1e-10)
    assert np.allclose(x_f, lasso_cvxpy(A, y, lam), atol=1e-3)
    assert np.allclose(x_f, lasso_sklearn(A, y, lam), atol=1e-3)


def test_fista_batch_equals_one_by_one():
    A, _, y1 = _random_problem(seed=0)
    _, _, y2 = _random_problem(seed=1)
    Y = np.stack([y1, A @ np.ones(A.shape[1]) * 0.01], axis=1)
    X = fista(A, Y, 0.01, n_iter=500)
    assert np.allclose(X[:, 0], fista(A, Y[:, 0], 0.01, n_iter=500))
    assert np.allclose(X[:, 1], fista(A, Y[:, 1], 0.01, n_iter=500))


def test_basis_pursuit_recovers_sparse_vector():
    A, x, y = _random_problem(m=60, n=200, s=5)
    assert np.allclose(basis_pursuit_cvxpy(A, y), x, atol=1e-4)


def test_extended_l1_finds_sparse_error():
    rng = np.random.default_rng(2)
    A, x, y = _random_problem(m=100, n=150, s=3)
    e = np.zeros(100)
    hit = rng.choice(100, 8, replace=False)
    e[hit] = 2.0
    x_hat, e_hat = extended_l1(A, y + e, lam=0.01, n_iter=3000)
    assert set(np.argsort(-np.abs(e_hat))[:8]) == set(hit)
    assert np.linalg.norm(x_hat - x) < 0.3 * np.linalg.norm(x)


def test_rpca_separates_low_rank_and_sparse():
    rng = np.random.default_rng(3)
    L0 = rng.standard_normal((100, 3)) @ rng.standard_normal((3, 60))
    S0 = np.zeros_like(L0)
    mask = rng.random(L0.shape) < 0.05
    S0[mask] = rng.choice([-10.0, 10.0], mask.sum())
    L, S, info = rpca(L0 + S0)
    assert info["rank"] == 3
    assert np.linalg.norm(L - L0) / np.linalg.norm(L0) < 1e-3
    assert np.linalg.norm(S - S0) / np.linalg.norm(S0) < 1e-3


def test_sci_extremes():
    masks = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=bool)
    concentrated = np.array([[1.0], [0.5], [0.0], [0.0]])
    spread = np.array([[1.0], [0.0], [1.0], [0.0]])
    assert np.isclose(sparsity_concentration_index(concentrated, masks)[0], 1.0)
    assert np.isclose(sparsity_concentration_index(spread, masks)[0], 0.0)


def test_src_on_union_of_subspaces():
    # Each class lives on its own random 3-dimensional subspace.
    rng = np.random.default_rng(4)
    d, C, per_class = 80, 6, 15
    bases = [rng.standard_normal((d, 3)) for _ in range(C)]
    X = np.vstack([(B @ rng.standard_normal((3, per_class))).T for B in bases])
    labels = np.repeat(np.arange(C), per_class)
    X_test = np.vstack([(B @ rng.standard_normal(3)) for B in bases])
    clf = SRCClassifier(lam=0.001, n_iter=1000).fit(X, labels)
    pred, score = clf.predict_with_score(X_test)
    assert (pred == np.arange(C)).all()
    assert (score > 0.5).all()


def test_choose_tau_and_decisions():
    vis = np.array([0.1, 0.2, 0.3, 0.9])
    tau = choose_tau(vis, target_far=0.0)
    assert far_frr(tau, [0], [0], [0.95], vis) == (0.0, 0.0)
    dec = access_decision([1, 2, 3], [0.9, 0.9, 0.1], tau=0.5,
                          occluded_fraction=[0.0, 0.8, 0.0], rho=0.4, access_list=[1, 2])
    assert list(dec) == ["accept", "badge", "reject"]