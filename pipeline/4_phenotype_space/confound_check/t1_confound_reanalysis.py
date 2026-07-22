#!/usr/bin/env python3
"""
T1: Report-type confound reanalysis on 5-vote llama + GMM k=4 (v3.2).

Replicates singlevote analyses 4.1 (report-type confound identification) and
4.2 (covariate-controlled re-verification) on the CURRENT k4 input, and adds:
  (A) mode x Report_type crosstab (uses saved production GMM k=4 labels)
  (B) report-length covariate analysis (sentence length, word count, true txt
      length; correlation with PC1; regress-out alongside type; mode x length)
  Supplementary: Sex, ADOS Module crosstabs.

Conventions:
  - 4.1 / 4.2 replication: ILR + sklearn-centered (covariance) PCA on ALL 489
    reports, comparisons on ASD subset. v2 (2026-06-02): switched z-scored ->
    sklearn-centered to unify with the rest of the pipeline (old z-scored in
    old_zscored/). Core confound results (mode x type V, regress-out) are
    PCA-convention-independent; only PC1 x type r / eff_dim are PCA-derived.
  - Mode-related parts (crosstab, GMM stability ARI): PRODUCTION convention
    (ASD-only ILR -> sklearn PCA -> GMM k=4 on PC1-3, seed 42, n_init 5),
    matching step5 mode_discovery saved labels.

Outputs JSON + console to this script's own folder (1_confound_check/).
"""

import json
import os
import warnings
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── paths ──────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get("REPORT_LLM_PIPELINE_ROOT", REPO_ROOT / "pipeline"))
DATA_DIR = Path(os.environ.get(
    "DOMAIN_VECTOR_OUTPUT_DIR",
    REPO_ROOT / "pipeline" / "3_multidomain_vectors" / "3_2_domain_frequency_vector" / "outputs",
))
DOMAIN_TSV = DATA_DIR / "domain_vectors_proportion_latest.tsv"
DOMAIN_META = DATA_DIR / "domain_vectors_meta_latest.json"
# mode/cluster labels from the raw step5 output (fixed filename; run step5 first)
CLUSTER_TSV = (Path(__file__).resolve().parent.parent /
               "3_raw/step5_mode_discovery/mode_discovery/all_members_with_cluster_and_mahal.tsv")
# report corpus (489)
REPORT_TXT_DIR = Path(os.environ.get(
    "REPORT_TXT_DIR",
    REPO_ROOT / "pipeline" / "1_classifier" / "data" / "489reports" /
    "reports_txt_no_headings",
))

COVAR_CSV = Path(os.environ.get(
    "ATTENTION_COVARIATE_CSV",
    REPO_ROOT / "pipeline" / "3_multidomain_vectors" / "3_1_top_10_sentences" /
    "outputs" / "group_analysis" / "merged_attention_with_keywords.csv",
))
OUT_DIR = Path(__file__).resolve().parent

N_PCS = 3
SEED = 42
PSEUDOCOUNT = 1e-6
K_GMM = 4
CLUSTER_LABELS = {0: "M1 prototype", 1: "M2 prototype",
                  2: "M3 diffuse_cloud", 3: "M4 directional"}

# ── ILR / PCA helpers ──────────────────────────────────────────────
def helmert_basis(D):
    V = np.zeros((D, D - 1))
    for j in range(D - 1):
        scale = np.sqrt((j + 1) / (j + 2))
        V[:j + 1, j] = scale / (j + 1)
        V[j + 1, j] = -scale
    return V

def ilr_transform(X, pseudocount=PSEUDOCOUNT):
    X_adj = X + pseudocount
    X_adj = X_adj / X_adj.sum(axis=1, keepdims=True)
    return np.log(X_adj) @ helmert_basis(X_adj.shape[1]), helmert_basis(X_adj.shape[1])

def pca_centered(X_ilr):

    from sklearn.decomposition import PCA
    pca = PCA(n_components=X_ilr.shape[1], random_state=42)
    scores = pca.fit_transform(X_ilr)
    eigvals = pca.explained_variance_
    explained = pca.explained_variance_ratio_
    props = eigvals[eigvals > 0] / eigvals.sum()
    eff_dim = float(np.exp(-np.sum(props * np.log(props))))
    n_kaiser = int((eigvals >= eigvals.mean()).sum())  # Kaiser-Guttman (covariance PCA)
    return dict(scores=scores, eigvals=eigvals, explained=explained,
                eff_dim=eff_dim, n_kaiser=n_kaiser)

def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 +
                      (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0

def bh_fdr(pvals, q=0.10):
    p = np.asarray(pvals)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * (np.arange(1, n + 1)) / n
    passed = ranked <= thresh
    sig = np.zeros(n, dtype=bool)
    if passed.any():
        kmax = np.max(np.where(passed)[0])
        sig_order = order[:kmax + 1]
        sig[sig_order] = True
    return sig

def heavy_tail_test(X):
    n, p = X.shape
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    cov_inv = np.linalg.inv(cov)
    diff = X - mu
    d2 = np.sum(diff @ cov_inv * diff, axis=1)
    beta2 = float(np.mean(d2 ** 2))
    expected = p * (p + 2)
    z = float((beta2 - expected) / np.sqrt(8 * p * (p + 2) / n))
    chi2_99 = stats.chi2.ppf(0.99, df=p)
    excess_99 = float(np.percentile(d2, 99) / chi2_99)
    return dict(beta2=beta2, expected=int(expected), z_stat=z,
                heavy_tailed=bool(z > 1.96), excess_99=excess_99,
                d2_mean=float(d2.mean()),
                d2_kurtosis=float(stats.kurtosis(d2, fisher=True)))

def deviation_direction_test(X, n_perm=2000, rng=None):
    """Top-10% Mahalanobis outliers: pairwise cosine vs permutation null."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    n, p = X.shape
    mu = X.mean(axis=0)
    cov_inv = np.linalg.inv(np.cov(X, rowvar=False))
    diff = X - mu
    mahal = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
    thr = np.percentile(mahal, 90)
    mask = mahal >= thr
    k = int(mask.sum())
    dev = diff[mask]
    norms = np.linalg.norm(dev, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1e-10
    unit = dev / norms
    obs = float(np.mean([unit[i] @ unit[j] for i, j in combinations(range(k), 2)]))
    perm = []
    for _ in range(n_perm):
        idx = rng.choice(n, size=k, replace=False)
        pd_ = diff[idx]
        pn = np.linalg.norm(pd_, axis=1, keepdims=True)
        pn[pn < 1e-10] = 1e-10
        pu = pd_ / pn
        perm.append(np.mean([pu[i] @ pu[j] for i, j in combinations(range(k), 2)]))
    perm = np.array(perm)
    p_value = float((perm >= obs).mean())
    null_95 = float(np.percentile(perm, 95))
    verdict = "DIVERSE_DIRECTION" if obs <= null_95 else "SHARED_DIRECTION"
    return dict(n_outliers=k, threshold=float(thr), observed_mean_cosine=obs,
                null_95th=null_95, null_mean=float(perm.mean()),
                p_value=p_value, verdict=verdict)

def production_gmm_labels(X_asd_props):
    """Reproduce step5 production labels: ASD-only ILR -> sklearn PCA -> GMM k4."""
    X_ilr, _ = ilr_transform(X_asd_props)
    pca = PCA(n_components=X_ilr.shape[1], random_state=SEED)
    X_pca = pca.fit_transform(X_ilr)
    gmm = GaussianMixture(n_components=K_GMM, covariance_type="full",
                          n_init=5, random_state=SEED)
    gmm.fit(X_pca[:, :N_PCS])
    return gmm.predict(X_pca[:, :N_PCS]), X_pca

def gmm_on_residual(X_resid_asd):
    """GMM k4 on residual ASD-only PC1-3 (production-style)."""
    pca = PCA(n_components=X_resid_asd.shape[1], random_state=SEED)
    X_pca = pca.fit_transform(X_resid_asd)
    gmm = GaussianMixture(n_components=K_GMM, covariance_type="full",
                          n_init=5, random_state=SEED)
    gmm.fit(X_pca[:, :N_PCS])
    labels = gmm.predict(X_pca[:, :N_PCS])
    sil = float(silhouette_score(X_pca[:, :N_PCS], labels))
    return labels, sil

# ── load data ──────────────────────────────────────────────────────
def load():
    df = pd.read_csv(DOMAIN_TSV, sep="\t")
    meta = json.load(open(DOMAIN_META))
    domains = meta["domain_columns"]
    report_ids = df["report_id"].values
    asd = df["asd_label"].values.astype(int)
    props = df[domains].values.astype(float)
    rtype = np.array([rid.split("-")[-1][0] for rid in report_ids])  # A / P

    # cluster labels (ASD only, 346)
    cl = pd.read_csv(CLUSTER_TSV, sep="\t")
    cl_map = dict(zip(cl["report_id"], cl["cluster"]))

    # covariates: aggregate sentence-level -> report-level.

    if COVAR_CSV.exists():
        cov = pd.read_csv(COVAR_CSV)
        agg = cov.groupby("report_id").agg(
            sent_len_mean=("sentence_length", "mean"),
            sent_len_sum=("sentence_length", "sum"),
            word_count_mean=("word_count", "mean"),
            word_count_sum=("word_count", "sum"),
            n_sent=("sentence_idx", "count"),
            Sex=("Sex", "first"),
            Module=("Module", "first"),
            Report_type=("Report_type", "first"),
        ).reset_index()
        cov_map = agg.set_index("report_id").to_dict("index")
    else:
        print(f"[warn] COVAR_CSV not found ({COVAR_CSV}); attention covariates will be NaN. "
              "Create the covariate CSV from the 3.1 preprocess/additional analysis if needed.")
        cov_map = {}

    # true txt length
    txt_len = {}
    for rid in report_ids:
        f = REPORT_TXT_DIR / f"{rid}.txt"
        if f.exists():
            t = f.read_text(errors="ignore")
            txt_len[rid] = (len(t), len(t.split()))
        else:
            txt_len[rid] = (np.nan, np.nan)

    return dict(df=df, domains=domains, report_ids=report_ids, asd=asd,
                props=props, rtype=rtype, cl_map=cl_map, cov_map=cov_map,
                txt_len=txt_len)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rng = np.random.default_rng(SEED)
    D = load()
    domains = D["domains"]
    props = D["props"]
    asd_mask = D["asd"] == 1
    rtype = D["rtype"]
    n = len(props)
    results = {"timestamp": ts, "version": "5vote_llama_k4_v3.2"}

    asd_A = (rtype == "A") & asd_mask
    asd_P = (rtype == "P") & asd_mask
    print(f"N={n} (A={int((rtype=='A').sum())}, P={int((rtype=='P').sum())}); "
          f"ASD={int(asd_mask.sum())} (A={int(asd_A.sum())}, P={int(asd_P.sum())})")
    results["counts"] = dict(n=n, n_A=int((rtype == "A").sum()),
                             n_P=int((rtype == "P").sum()),
                             n_asd=int(asd_mask.sum()),
                             n_asd_A=int(asd_A.sum()), n_asd_P=int(asd_P.sum()))

    # ILR + sklearn-centered (covariance) PCA on ALL 489 (4.1/4.2 convention, v2)
    X_ilr, V = ilr_transform(props)
    pca_all = pca_centered(X_ilr)
    scores = pca_all["scores"]

    # ══ 4.1 domain comparison (ASD only) ══
    print("\n=== 4.1 Domain profile A vs P (ASD only) ===")
    dom_rows, pvals = [], []
    for i, dom in enumerate(domains):
        a, b = props[asd_A, i], props[asd_P, i]
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        d = cohens_d(a, b)
        dom_rows.append(dict(domain=dom, mean_A=float(a.mean()),
                             mean_P=float(b.mean()), diff=float(a.mean() - b.mean()),
                             cohens_d=d, mann_whitney_p=float(p)))
        pvals.append(p)
    sig_fdr = bh_fdr(pvals, q=0.10)
    for r, s in zip(dom_rows, sig_fdr):
        r["fdr_sig_q0.10"] = bool(s)
    n_raw = sum(r["mann_whitney_p"] < 0.05 for r in dom_rows)
    n_fdr = int(sig_fdr.sum())
    print(f"  significant raw p<0.05: {n_raw}/{len(domains)}; BH-FDR q<=0.10: {n_fdr}")
    results["domain_comparison"] = dict(n_sig_raw=n_raw, n_sig_fdr=n_fdr,
                                        rows=dom_rows)

    # ══ 4.1 PC x type ══
    print("\n=== 4.1 PC scores x report type (ASD only) ===")
    type_bin_A = (rtype == "A").astype(int)  # A=1, P=0 (as original)
    pc_rows = []
    for k in range(N_PCS):
        a, b = scores[asd_A, k], scores[asd_P, k]
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        d = cohens_d(a, b)
        r_pb, p_pb = stats.pointbiserialr(type_bin_A[asd_mask], scores[asd_mask, k])
        pc_rows.append(dict(pc=f"PC{k+1}", cohens_d=d, mann_whitney_p=float(p),
                            pointbiserial_r=float(r_pb), r2=float(r_pb ** 2),
                            pb_p=float(p_pb)))
        print(f"  PC{k+1}: d={d:+.3f} p={p:.3e} | point-biserial r={r_pb:+.3f} "
              f"r2={r_pb**2:.3f}")
    results["pc_type"] = pc_rows

    # ══ 4.1 Mahalanobis (17D ILR) x type ══
    mu = X_ilr.mean(axis=0)
    cov_inv = np.linalg.inv(np.cov(X_ilr, rowvar=False))
    mahal = np.sqrt(np.sum((X_ilr - mu) @ cov_inv * (X_ilr - mu), axis=1))
    mA, mP = mahal[asd_A], mahal[asd_P]
    u_m, p_m = stats.mannwhitneyu(mA, mP, alternative="two-sided")
    results["mahalanobis_type"] = dict(A_mean=float(mA.mean()), P_mean=float(mP.mean()),
                                       cohens_d=cohens_d(mA, mP), mann_whitney_p=float(p_m))
    print(f"\n=== 4.1 Mahalanobis x type: d={cohens_d(mA,mP):+.3f} p={p_m:.3e} ===")

    # ══ 4.1 heavy-tail by type ══
    ht_by = {t: heavy_tail_test(X_ilr[m]) for t, m in [("A", asd_A), ("P", asd_P)]}
    results["heavy_tail_by_type"] = ht_by
    print(f"  heavy-tail A z={ht_by['A']['z_stat']:.1f}, P z={ht_by['P']['z_stat']:.1f}")

    # ══ 4.2 regress out type ══
    print("\n=== 4.2 Regress out report type from ILR ===")
    covariate = (rtype == "P").astype(float)  # A=0, P=1
    design = np.column_stack([np.ones(n), covariate])
    resid = np.zeros_like(X_ilr)
    r2_dim = []
    for j in range(X_ilr.shape[1]):
        beta = np.linalg.lstsq(design, X_ilr[:, j], rcond=None)[0]
        resid[:, j] = X_ilr[:, j] - design @ beta
        ss_tot = np.sum((X_ilr[:, j] - X_ilr[:, j].mean()) ** 2)
        r2_dim.append(1 - np.sum(resid[:, j] ** 2) / ss_tot if ss_tot > 0 else 0)
    mean_r2, max_r2 = float(np.mean(r2_dim)), float(np.max(r2_dim))
    print(f"  type explains ILR variance: mean R2={mean_r2:.4f} max R2={max_r2:.4f}")

    pca_resid = pca_centered(resid)
    r_before, _ = stats.pointbiserialr(covariate[asd_mask], scores[asd_mask, 0])
    r_after, _ = stats.pointbiserialr(covariate[asd_mask], pca_resid["scores"][asd_mask, 0])
    print(f"  PC1 x type r: before {r_before:+.3f} -> after {r_after:+.3f}")

    ht_all_b, ht_all_a = heavy_tail_test(X_ilr), heavy_tail_test(resid)
    ht_asd_b = heavy_tail_test(X_ilr[asd_mask])
    ht_asd_a = heavy_tail_test(resid[asd_mask])
    dev_b = deviation_direction_test(X_ilr[asd_mask], rng=np.random.default_rng(SEED))
    dev_a = deviation_direction_test(resid[asd_mask], rng=np.random.default_rng(SEED))
    print(f"  heavy-tail all: z {ht_all_b['z_stat']:.1f} -> {ht_all_a['z_stat']:.1f}")
    print(f"  heavy-tail ASD: z {ht_asd_b['z_stat']:.1f} -> {ht_asd_a['z_stat']:.1f}")
    print(f"  eff_dim: {pca_all['eff_dim']:.2f} -> {pca_resid['eff_dim']:.2f}")
    print(f"  deviation: {dev_b['verdict']}(p={dev_b['p_value']:.3f}) -> "
          f"{dev_a['verdict']}(p={dev_a['p_value']:.3f})")

    results["control_4_2"] = dict(
        mean_r2=mean_r2, max_r2=max_r2,
        pc1_type_r_before=float(r_before), pc1_type_r_after=float(r_after),
        heavy_tail_all_before=ht_all_b, heavy_tail_all_after=ht_all_a,
        heavy_tail_asd_before=ht_asd_b, heavy_tail_asd_after=ht_asd_a,
        eff_dim_before=pca_all["eff_dim"], eff_dim_after=pca_resid["eff_dim"],
        n_kaiser_before=pca_all["n_kaiser"], n_kaiser_after=pca_resid["n_kaiser"],
        pc1_var_before=float(pca_all["explained"][0]),
        pc1_var_after=float(pca_resid["explained"][0]),
        deviation_before=dev_b, deviation_after=dev_a)

    # ══ 4.2 GMM mode stability (production convention, ASD-only) ══
    print("\n=== 4.2 GMM k=4 mode stability (ASD-only, residual vs raw) ===")
    asd_props = props[asd_mask]
    prod_labels, _ = production_gmm_labels(asd_props)
    # baseline reproduction check vs saved labels
    saved = np.array([D["cl_map"].get(rid, -1)
                      for rid in D["report_ids"][asd_mask]])
    ari_repro = float(adjusted_rand_score(saved, prod_labels))
    # residual GMM
    resid_asd = resid[asd_mask]
    resid_labels, sil_resid = gmm_on_residual(resid_asd)
    ari = float(adjusted_rand_score(prod_labels, resid_labels))
    nmi = float(normalized_mutual_info_score(prod_labels, resid_labels))
    print(f"  reproduce saved labels ARI={ari_repro:.3f}")
    print(f"  raw vs residual GMM: ARI={ari:.3f} NMI={nmi:.3f} sil_resid={sil_resid:.3f}")
    print(f"  raw sizes {np.bincount(prod_labels).tolist()} | "
          f"resid sizes {np.bincount(resid_labels).tolist()}")
    results["gmm_stability"] = dict(
        ari_reproduce_saved=ari_repro, ari_raw_vs_resid=ari, nmi_raw_vs_resid=nmi,
        sil_resid=sil_resid,
        raw_sizes=np.bincount(prod_labels).tolist(),
        resid_sizes=np.bincount(resid_labels).tolist())

    # ══ ADD-A mode x Report_type crosstab (saved labels) ══
    print("\n=== ADD-A mode x Report_type crosstab (saved GMM k=4 labels) ===")
    asd_ids = D["report_ids"][asd_mask]
    asd_clusters = np.array([D["cl_map"].get(rid, -1) for rid in asd_ids])
    asd_rtype = rtype[asd_mask]
    ct = pd.crosstab(pd.Series(asd_clusters, name="mode"),
                     pd.Series(asd_rtype, name="type"))
    chi2, p_ct, dof, _ = stats.chi2_contingency(ct)
    cramers_v = float(np.sqrt(chi2 / (ct.values.sum() *
                                      (min(ct.shape) - 1))))
    print(ct)
    print(f"  chi2={chi2:.3f} p={p_ct:.4e} dof={dof} Cramer's V={cramers_v:.3f}")
    ct_pct = ct.div(ct.sum(axis=1), axis=0)
    results["mode_type_crosstab"] = dict(
        table={f"mode{m}": {t: int(ct.loc[m, t]) for t in ct.columns}
               for m in ct.index},
        pct_P_within_mode={f"mode{m}": float(ct_pct.loc[m, "P"])
                           if "P" in ct_pct.columns else 0.0 for m in ct.index},
        chi2=float(chi2), p=float(p_ct), dof=int(dof), cramers_v=cramers_v,
        labels=CLUSTER_LABELS)

    # ══ ADD-B length covariate ══
    print("\n=== ADD-B length covariate ===")
    cov_map = D["cov_map"]
    txt_len = D["txt_len"]
    rids_all = D["report_ids"]
    sent_len_mean = np.array([cov_map.get(r, {}).get("sent_len_mean", np.nan) for r in rids_all])
    word_sum = np.array([cov_map.get(r, {}).get("word_count_sum", np.nan) for r in rids_all])
    txt_words = np.array([txt_len[r][1] for r in rids_all], dtype=float)

    def corr_pc1(x, label):
        m = asd_mask & np.isfinite(x)
        if int(m.sum()) < 3:
            print(f"  PC1 x {label}: n={int(m.sum())} (insufficient — skipped, NaN)")
            return dict(pearson_r=None, pearson_p=None,
                        spearman_r=None, spearman_p=None, n=int(m.sum()))
        r, p = stats.pearsonr(x[m], scores[m, 0])
        rs, ps = stats.spearmanr(x[m], scores[m, 0])
        print(f"  PC1 x {label}: pearson r={r:+.3f}(p={p:.3f}) "
              f"spearman={rs:+.3f}(p={ps:.3f}) n={int(m.sum())}")
        return dict(pearson_r=float(r), pearson_p=float(p),
                    spearman_r=float(rs), spearman_p=float(ps), n=int(m.sum()))

    len_pc1 = dict(sent_len_mean=corr_pc1(sent_len_mean, "sent_len_mean"),
                   word_count_sum=corr_pc1(word_sum, "word_count_sum"),
                   txt_words=corr_pc1(txt_words, "txt_true_words"))

    # mode x length (Kruskal-Wallis over 4 modes, ASD only)
    def kw_by_mode(x, label):
        groups = []
        for c in range(K_GMM):
            v = x[asd_mask][(asd_clusters == c) & np.isfinite(x[asd_mask])]
            groups.append(v)
        nonempty = [g for g in groups if len(g) > 0]
        means = [float(np.nanmean(g)) if len(g) else None for g in groups]
        if len(nonempty) < 2:
            print(f"  mode x {label}: insufficient — skipped (NaN)")
            return dict(H=None, p=None, mode_means=means)
        H, p = stats.kruskal(*nonempty)
        print(f"  mode x {label}: KW H={H:.3f} p={p:.4f} means={[round(m,1) if m is not None else None for m in means]}")
        return dict(H=float(H), p=float(p), mode_means=means)

    len_mode = dict(sent_len_mean=kw_by_mode(sent_len_mean, "sent_len_mean"),
                    word_count_sum=kw_by_mode(word_sum, "word_count_sum"),
                    txt_words=kw_by_mode(txt_words, "txt_true_words"))

    # regress out type + length jointly; recheck PC1 x type
    Lz = (word_sum - np.nanmean(word_sum)) / np.nanstd(word_sum)
    Lz = np.nan_to_num(Lz)
    design2 = np.column_stack([np.ones(n), covariate, Lz])
    resid2 = np.zeros_like(X_ilr)
    for j in range(X_ilr.shape[1]):
        beta = np.linalg.lstsq(design2, X_ilr[:, j], rcond=None)[0]
        resid2[:, j] = X_ilr[:, j] - design2 @ beta
    pca_r2 = pca_centered(resid2)
    r_after2, _ = stats.pointbiserialr(covariate[asd_mask], pca_r2["scores"][asd_mask, 0])
    print(f"  PC1 x type after type+length control: r={r_after2:+.3f}")
    results["length_covariate"] = dict(pc1_corr=len_pc1, mode_kruskal=len_mode,
                                       pc1_type_r_after_type_plus_length=float(r_after2))

    # ══ Supplementary: Sex, Module crosstab vs mode ══
    print("\n=== Supplementary: mode x Sex, mode x ADOS Module ===")
    sex = np.array([str(cov_map.get(r, {}).get("Sex", "NA")) for r in rids_all])
    module = np.array([str(cov_map.get(r, {}).get("Module", "NA")) for r in rids_all])
    supp = {}
    for name, arr in [("Sex", sex[asd_mask]), ("Module", module[asd_mask])]:
        ctx = pd.crosstab(pd.Series(asd_clusters, name="mode"),
                          pd.Series(arr, name=name))
        # drop NA/empty cols for chi2
        ctx_clean = ctx.loc[:, [c for c in ctx.columns if c not in ("NA", "nan", "")]]
        try:
            c2, pp, dd, _ = stats.chi2_contingency(ctx_clean)
            cv = float(np.sqrt(c2 / (ctx_clean.values.sum() * (min(ctx_clean.shape) - 1))))
        except Exception:
            c2, pp, dd, cv = float("nan"), float("nan"), 0, float("nan")
        print(f"  mode x {name}: chi2={c2:.3f} p={pp:.4f} V={cv:.3f}")
        supp[name] = dict(chi2=float(c2), p=float(pp), dof=int(dd), cramers_v=cv,
                          table={f"mode{m}": {c: int(ctx.loc[m, c]) for c in ctx.columns}
                                 for m in ctx.index})
    results["supplementary_crosstabs"] = supp

    # save
    out = OUT_DIR / f"t1_confound_results.json"
    json.dump(results, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"\n[saved] {out}")

if __name__ == "__main__":
    main()
