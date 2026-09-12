# -*- coding: utf-8 -*-

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy import stats
from scipy.special import digamma, gammaln

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get(
    "DOMAIN_VECTOR_OUTPUT_DIR",
    REPO_ROOT / "pipeline" / "3_multidomain_vectors" / "domain_frequency_vector" / "outputs",
))
PROPORTION_TSV = DATA_DIR / "domain_vectors_proportion_latest.tsv"
META_JSON = DATA_DIR / "domain_vectors_meta_latest.json"

SEED = 42
PSEUDOCOUNT = 1e-6

def helmert(D):
    V = np.zeros((D, D - 1))
    for i in range(D - 1):
        V[:i + 1, i] = 1.0 / (i + 1)
        V[i + 1, i] = -1.0
        V[:, i] *= np.sqrt((i + 1.0) / (i + 2.0))
    return V

def clr(X):
    Xp = X + PSEUDOCOUNT
    Xp = Xp / Xp.sum(1, keepdims=True)
    L = np.log(Xp)
    return L - L.mean(1, keepdims=True)

def ilr(X):
    return clr(X) @ helmert(X.shape[1])

def residualize(M, design):
    R = np.zeros_like(M)
    for j in range(M.shape[1]):
        b = np.linalg.lstsq(design, M[:, j], rcond=None)[0]
        R[:, j] = M[:, j] - design @ b
    return R

def variance_partition(X, labels):
    g = X.mean(0)
    total = ((X - g) ** 2).sum()
    between = sum(np.sum(labels == k) * ((X[labels == k].mean(0) - g) ** 2).sum()
                 for k in np.unique(labels))
    return total, between, total - between, between / total

def load():
    """Return the canonical type-residual representation + raw helpers (ASD only)."""
    df = pd.read_csv(PROPORTION_TSV, sep="\t")
    dom = json.load(open(META_JSON))["domain_columns"]
    sub = df[df["asd_label"].values == 1].reset_index(drop=True)
    rid = sub["report_id"].values
    P = sub[dom].values.astype(float)
    rtype = np.array([r.split("-")[-1][0] for r in rid])
    design = np.column_stack([np.ones(len(P)), (rtype == "P").astype(float)])
    ILR = ilr(P)
    RES = residualize(ILR, design)
    CLR_RES = residualize(clr(P), design)
    return dict(rid=rid, dom=dom, P=P, rtype=rtype,
                n_A=int((rtype == "A").sum()), n_P=int((rtype == "P").sum()),
                ILR=ILR, RES=RES, CLR_RES=CLR_RES, helmert=helmert(P.shape[1]))

def pca_scores(M, k_lead=3):
    p = PCA(n_components=M.shape[1], random_state=SEED).fit(M)
    s = p.transform(M)
    return p, s, s[:, :k_lead], p.explained_variance_ratio_

def gmm_fit(lead, k):
    g = GaussianMixture(n_components=k, covariance_type="full", n_init=10,
                        random_state=SEED, max_iter=500).fit(lead)
    return g.predict(lead), g.predict_proba(lead), g

# heavy-tail / multivariate-t helpers
def mardia_z(X):
    """Multivariate kurtosis (Mardia) z-score + 99th-percentile excess (quantile ratio)."""
    n, p = X.shape
    mu = X.mean(0)
    ci = np.linalg.inv(np.cov(X, rowvar=False))
    d2 = np.sum((X - mu) @ ci * (X - mu), 1)
    beta2 = np.mean(d2 ** 2)
    exp = p * (p + 2)
    z = (beta2 - exp) / np.sqrt(8 * p * (p + 2) / n)
    chi99 = stats.chi2.ppf(0.99, df=p)
    exc = np.percentile(d2, 99) / chi99
    return dict(p=int(p), beta2=float(beta2), expected=int(exp), z=float(z),
                excess99=float(exc), kurt="lepto" if z > 0 else "platy")

def t_loglik(X, mu, S, nu):
    n, p = X.shape
    d = X - mu
    _, ld = np.linalg.slogdet(S)
    Si = np.linalg.inv(S)
    m2 = np.sum(d @ Si * d, 1)
    return np.sum(gammaln((nu + p) / 2) - gammaln(nu / 2) - (p / 2) * np.log(nu * np.pi)
                  - 0.5 * ld - ((nu + p) / 2) * np.log(1 + m2 / nu))

def fit_t(X):
    """EM fit of a multivariate-t. Returns (nu, loglik, n_params)."""
    n, p = X.shape
    mu = X.mean(0)
    S = np.cov(X, rowvar=False)
    nu = 5.0
    prev = -np.inf
    for _ in range(300):
        d = X - mu
        Si = np.linalg.inv(S)
        m2 = np.sum(d @ Si * d, 1)
        w = (nu + p) / (nu + m2)
        ws = w.sum()
        mu = (w[:, None] * X).sum(0) / ws
        dn = X - mu
        S = (dn * w[:, None]).T @ dn / n
        S = (S + S.T) / 2
        if np.linalg.eigvalsh(S).min() < 1e-10:
            S += np.eye(p) * 1e-6

        def eq(v):
            return (-digamma(v / 2) + np.log(v / 2) + 1 + (1 / n) * np.sum(np.log(w) - w)
                    + digamma((v + p) / 2) - np.log((v + p) / 2))
        lo, hi = 0.1, 500.0
        for _ in range(100):
            m = (lo + hi) / 2
            if eq(m) > 0:
                lo = m
            else:
                hi = m
            if hi - lo < 0.01:
                break
        nu = (lo + hi) / 2
        ll = t_loglik(X, mu, S, nu)
        if abs(ll - prev) < 1e-6:
            break
        prev = ll
    return nu, ll, p + p * (p + 1) // 2 + 1
