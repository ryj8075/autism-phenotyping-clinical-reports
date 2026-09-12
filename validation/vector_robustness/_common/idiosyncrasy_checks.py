from __future__ import annotations

from itertools import combinations
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.special import digamma, gammaln
from sklearn.mixture import GaussianMixture

# ILR transform
def _helmert_basis(D: int) -> np.ndarray:
    V = np.zeros((D, D - 1))
    for j in range(D - 1):
        scale = np.sqrt((j + 1) / (j + 2))
        V[: j + 1, j] = scale / (j + 1)
        V[j + 1, j] = -scale
    return V

def ilr_transform(X: np.ndarray, pseudocount: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    X_adj = X + pseudocount
    X_adj = X_adj / X_adj.sum(axis=1, keepdims=True)
    log_X = np.log(X_adj)
    V = _helmert_basis(X_adj.shape[1])
    return log_X @ V, V

def residualize_on_ptype(X_ilr: np.ndarray, ptype: np.ndarray) -> np.ndarray:
    ptype = np.asarray(ptype)
    is_p = (ptype == "P").astype(float)
    Z = np.column_stack([np.ones(len(is_p)), is_p])  # (N, 2)
    beta, *_ = np.linalg.lstsq(Z, X_ilr, rcond=None)
    return X_ilr - Z @ beta

# A. Heavy-tail test (Mardia)
def heavy_tail_test(X: np.ndarray) -> Dict:
    n, p = X.shape
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    cov_inv = np.linalg.inv(cov)
    diff = X - mu
    d2 = np.sum(diff @ cov_inv * diff, axis=1)

    beta2 = np.mean(d2 ** 2)
    expected = p * (p + 2)
    z = (beta2 - expected) / np.sqrt(8 * p * (p + 2) / n)

    chi2_99 = stats.chi2.ppf(0.99, df=p)
    obs_99 = np.percentile(d2, 99)
    excess_99 = obs_99 / chi2_99

    ks_stat, ks_p = stats.kstest(d2, "chi2", args=(p,))

    return {
        "beta2": float(beta2),
        "expected": int(expected),
        "z_stat": float(z),
        "heavy_tailed": bool(z > 1.96),
        "excess_99": float(excess_99),
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
        "d2_mean": float(d2.mean()),
        "d2_kurtosis": float(stats.kurtosis(d2, fisher=True)),
    }

# B. PCA / eigenvalue structure
def pca_analysis(X: np.ndarray) -> Dict:
    Xc = X - X.mean(axis=0)
    cov_mat = np.cov(Xc, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    explained = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(explained)
    n_kaiser = int((eigenvalues >= eigenvalues.mean()).sum())

    # Effective dimensionality (Shannon entropy of eigenvalue distribution)
    props_nz = explained[explained > 0]
    H = -np.sum(props_nz * np.log(props_nz))
    eff_dim = float(np.exp(H))

    scores = Xc @ eigenvectors

    return {
        "eigenvalues": eigenvalues.tolist(),
        "explained_variance_ratio": explained.tolist(),
        "cumulative": cumulative.tolist(),
        "n_kaiser": n_kaiser,
        "effective_dim": eff_dim,
        "pc1_variance": float(explained[0]),
        "pc2_variance": float(explained[1]) if len(explained) > 1 else None,
        "pc3_variance": float(explained[2]) if len(explained) > 2 else None,
        "scores": scores,
    }

# C. GMM vs multivariate t
def fit_t_distribution(X: np.ndarray, max_iter: int = 200,
                       tol: float = 1e-6) -> Tuple[float, float, float]:

    n, p = X.shape
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    nu = 10.0

    for _ in range(max_iter):
        diff = X - mu
        cov_inv = np.linalg.inv(cov)
        d2 = np.sum(diff @ cov_inv * diff, axis=1)
        w = (nu + p) / (nu + d2)

        w_sum = w.sum()
        mu_new = (w[:, None] * X).sum(axis=0) / w_sum
        diff_new = X - mu_new
        cov_new = (w[:, None, None] *
                   (diff_new[:, :, None] * diff_new[:, None, :])).sum(axis=0) / w_sum

        def nu_obj(v):
            return (-digamma(v / 2) + np.log(v / 2) + 1 +
                    (np.log(w) - w).mean() + digamma((v + p) / 2) -
                    np.log((v + p) / 2))

        lo, hi = 0.1, 500
        for _ in range(50):
            mid = (lo + hi) / 2
            if nu_obj(mid) > 0:
                lo = mid
            else:
                hi = mid
        nu_new = (lo + hi) / 2

        if abs(nu_new - nu) < tol and np.max(np.abs(mu_new - mu)) < tol:
            break
        mu, cov, nu = mu_new, cov_new, nu_new

    sign, logdet = np.linalg.slogdet(cov)
    diff = X - mu
    cov_inv = np.linalg.inv(cov)
    d2 = np.sum(diff @ cov_inv * diff, axis=1)
    ll = n * (gammaln((nu + p) / 2) - gammaln(nu / 2) -
              p / 2 * np.log(nu * np.pi) - 0.5 * logdet)
    ll += np.sum(-(nu + p) / 2 * np.log(1 + d2 / nu))

    n_params = p + p * (p + 1) // 2 + 1
    bic = -2 * ll + n_params * np.log(n)
    return float(nu), float(bic), float(ll)

def gmm_vs_t_test(X_pca: np.ndarray, k_range=range(2, 9), seed: int = 42) -> Dict:
    results = {}
    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              n_init=5, random_state=seed)
        gmm.fit(X_pca)
        results[f"GMM_k{k}"] = {"bic": float(gmm.bic(X_pca)), "k": k}

    nu, t_bic, t_ll = fit_t_distribution(X_pca)
    results["t_dist"] = {"bic": float(t_bic), "nu": float(nu), "ll": float(t_ll)}

    best_gmm_k = min(k_range, key=lambda k: results[f"GMM_k{k}"]["bic"])
    best_gmm_bic = results[f"GMM_k{best_gmm_k}"]["bic"]

    winner = "GMM" if best_gmm_bic < t_bic else "t_dist"
    delta_bic = best_gmm_bic - t_bic

    return {
        "results": results,
        "best_gmm_k": int(best_gmm_k),
        "best_gmm_bic": best_gmm_bic,
        "t_bic": t_bic,
        "t_nu": nu,
        "winner": winner,
        "delta_bic_gmm_minus_t": float(delta_bic),
    }

# D. Deviation direction
def deviation_direction_test(X: np.ndarray, n_perm: int = 1000,
                             tail_percentile: float = 90.0,
                             seed: int = 42) -> Dict:

    rng = np.random.default_rng(seed)
    n, p = X.shape
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    cov_inv = np.linalg.inv(cov)
    diff = X - mu
    d2 = np.sum(diff @ cov_inv * diff, axis=1)
    mahal = np.sqrt(d2)

    threshold = np.percentile(mahal, tail_percentile)
    outlier_mask = mahal >= threshold
    n_outliers = int(outlier_mask.sum())
    if n_outliers < 3:
        return {
            "n_outliers": n_outliers,
            "observed_mean_cosine": None,
            "verdict": "INSUFFICIENT_OUTLIERS",
        }

    dev = diff[outlier_mask]
    norms = np.linalg.norm(dev, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1e-10
    unit = dev / norms

    cosines = [float(unit[i] @ unit[j])
               for i, j in combinations(range(n_outliers), 2)]
    observed = float(np.mean(cosines))

    perm_means = []
    for _ in range(n_perm):
        idx = rng.choice(n, size=n_outliers, replace=False)
        pdev = diff[idx]
        pnorm = np.linalg.norm(pdev, axis=1, keepdims=True)
        pnorm[pnorm < 1e-10] = 1e-10
        pu = pdev / pnorm
        pc = [float(pu[i] @ pu[j]) for i, j in combinations(range(n_outliers), 2)]
        perm_means.append(np.mean(pc))
    perm_means = np.array(perm_means)
    p_value = float((perm_means >= observed).mean())
    null_95 = float(np.percentile(perm_means, 95))
    null_mean = float(perm_means.mean())

    verdict = "DIVERSE_DIRECTION" if observed <= null_95 else "SHARED_DIRECTION"

    return {
        "n_outliers": n_outliers,
        "threshold": float(threshold),
        "observed_mean_cosine": observed,
        "null_mean": null_mean,
        "null_95th": null_95,
        "p_value": p_value,
        "verdict": verdict,
    }

# All-in-one runner
def run_all_checks(
    X_proportion: np.ndarray,
    n_lead_pcs: int = 3,
    n_perm_direction: int = 1000,
    seed: int = 42,
    residualize_ptype: np.ndarray = None,
) -> Dict:

    X_ilr, V = ilr_transform(X_proportion)
    if residualize_ptype is not None:
        X_ilr = residualize_on_ptype(X_ilr, residualize_ptype)

    heavy = heavy_tail_test(X_ilr)
    pca = pca_analysis(X_ilr)
    scores = pca["scores"]

    pca_serializable = {k: v for k, v in pca.items() if k != "scores"}

    n_pcs_used = min(n_lead_pcs, scores.shape[1])
    gmm_t = gmm_vs_t_test(scores[:, :n_pcs_used], seed=seed)

    dev = deviation_direction_test(X_ilr, n_perm=n_perm_direction, seed=seed)

    return {
        "A_heavy_tail": heavy,
        "B_eigenvalue": pca_serializable,
        "C_gmm_vs_t": gmm_t,
        "D_deviation_direction": dev,

        "_arrays": {
            "X_ilr": X_ilr,
            "pca_scores": scores,
        },
    }
