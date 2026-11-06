"""NumPy-only Gaussian-mixture and low-rank latent-prior fitting."""

import dataclasses
from typing import Tuple

import numpy as np


@dataclasses.dataclass
class GMMParams:
    weights: np.ndarray  # (K,)
    means: np.ndarray  # (K,D)
    covs: np.ndarray  # (K,D,D)


def log_gaussian(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    D = x.shape[-1]
    L = np.linalg.cholesky(cov)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    y = np.linalg.solve(L, (x - mean).T).T
    quad = np.sum(y * y, axis=1)
    return -0.5 * (D * np.log(2.0 * np.pi) + logdet + quad)


def gmm_em_fit(
    X: np.ndarray, K: int, max_iter: int = 200, tol: float = 1e-6, reg: float = 1e-5, seed: int = 0
) -> GMMParams:
    N, D = X.shape
    rng = np.random.default_rng(seed)
    # k-means++ init: pick first mean randomly, then dist^2 weighted
    means = np.empty((K, D))
    means[0] = X[rng.integers(0, N)]
    d2 = np.full(N, np.inf)
    for k in range(1, K):
        d2 = np.minimum(d2, np.sum((X - means[k - 1]) ** 2, axis=1))
        probs = d2 / d2.sum()
        means[k] = X[rng.choice(N, p=probs)]
    covs = np.array([np.cov(X, rowvar=False) + reg * np.eye(D) for _ in range(K)])
    weights = np.full(K, 1.0 / K)

    prev_ll = -np.inf
    for it in range(max_iter):
        # E-step
        log_resp = np.stack(
            [np.log(weights[k] + 1e-12) + log_gaussian(X, means[k], covs[k]) for k in range(K)],
            axis=1,
        )  # (N,K)
        # log-sum-exp
        m = np.max(log_resp, axis=1, keepdims=True)
        resp = np.exp(log_resp - m)
        resp_sum = resp.sum(axis=1, keepdims=True)
        resp = resp / (resp_sum + 1e-12)
        ll = np.sum(m + np.log(resp_sum + 1e-12))

        # M-step
        Nk = resp.sum(axis=0) + 1e-12
        weights = Nk / N
        means = (resp.T @ X) / Nk[:, None]
        for k in range(K):
            Xc = X - means[k]
            covs[k] = (Xc.T @ (Xc * resp[:, [k]])) / Nk[k]
            # regularize
            covs[k].flat[:: D + 1] += reg

        if abs(ll - prev_ll) < tol * (1 + abs(prev_ll)):
            break
        prev_ll = ll

    return GMMParams(weights=weights, means=means, covs=covs)


def gmm_bic(X: np.ndarray, gmm: GMMParams) -> float:
    N, D = X.shape
    K = gmm.weights.shape[0]
    # compute log-likelihood under the mixture
    log_comp = []
    for k in range(K):
        log_comp.append(
            np.log(gmm.weights[k] + 1e-12) + log_gaussian(X, gmm.means[k], gmm.covs[k])
        )
    log_comp = np.stack(log_comp, axis=1)  # (N,K)
    m = np.max(log_comp, axis=1, keepdims=True)
    ll = np.sum(m + np.log(np.exp(log_comp - m).sum(axis=1, keepdims=True)))
    # number of parameters: weights(K-1) + means(KD) + covs(K*D*(D+1)/2)
    n_params = (K - 1) + K * D + K * (D * (D + 1) // 2)
    bic = -2 * ll + n_params * np.log(N)
    return float(bic)


def fit_best_gmm(X: np.ndarray, kmax: int, seed: int) -> tuple[int, GMMParams, float]:
    best_k, best_gmm, best_bic = None, None, float("inf")
    for K in range(1, kmax + 1):
        gmm = gmm_em_fit(X, K, seed=seed)
        bic = gmm_bic(X, gmm)
        print(f"[GMM] K={K}  BIC={bic:.1f}")
        if bic < best_bic:
            best_k, best_gmm, best_bic = K, gmm, bic
    return best_k, best_gmm, best_bic


def fit_ppca(X: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (U, Lambda, sigma2) such that Sigma ≈ sigma2 I + U diag(Lambda) U^T.
    Simple spectral fit via PCA eigenvalues: choose top-rank eigenpairs, set sigma2 to
    mean of remaining eigenvalues (Tipping & Bishop, 1999).
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    C = np.cov(Xc, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    if rank <= 0:
        # fallback to spherical
        sigma2 = float(np.mean(eigvals))
        return eigvecs[:, :0], np.zeros((0,)), sigma2
    U = eigvecs[:, :rank]
    if rank < len(eigvals):
        sigma2 = float(np.mean(eigvals[rank:]))
    else:
        sigma2 = 1e-6
    Lambda = np.maximum(eigvals[:rank] - sigma2, 1e-9)
    return U, Lambda, sigma2


def ppca_cov(U: np.ndarray, Lambda: np.ndarray, sigma2: float) -> np.ndarray:
    return sigma2 * np.eye(U.shape[0]) + (U * Lambda) @ U.T
