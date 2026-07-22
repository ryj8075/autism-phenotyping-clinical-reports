# -*- coding: utf-8 -*-
"""step3: is H4 (two-layer fingerprint) robust to dropping / down-weighting low-fidelity domains?

Layer 1 (prototypicality): full-covariance Mahalanobis on the residual PCA scores (identical to
step7_normative). For each variant we report mean/p90/max/top-decile AND, crucially, the rank
Spearman + top-decile overlap against the baseline_full19 ordering -- i.e. does removing/shrinking
weak domains preserve WHO is atypical? We also track the manuscript example patient
REPORT-066-A01 (baseline Mahalanobis ~6.18, top ~4%).

Layer 2 (normative structure): between/within variance decomposition of the residual space under
the stable 2-way (core-vs-periphery) GMM split (identical to step9_variance), with a permutation
null. Baseline between-fraction is ~14.8%.

NOTE: the Layer-1 rank agreement here is INTERNAL robustness to domain choice, NOT silver-vs-gold
validation. Silver-vs-gold patient ordering is separately weak (Table S14, rho=-0.34); per the
2026-07-07 guidance the fingerprint is framed as a candidate summary, so the claim tested here is
only "the structure survives dropping weak domains", not "individual extremity is validated".
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
import _sensitivity_common as sc               # noqa: E402
cc = sc.cc
EXAMPLE = "REPORT-066-A01"
N_PERM = 1000

def mahalanobis_full(RES):
    _, scores, lead, evr = cc.pca_scores(RES)
    mu = scores.mean(0)
    Sinv = np.linalg.inv(np.cov(scores, rowvar=False))
    m = np.sqrt(np.einsum("ij,jk,ik->i", scores - mu, Sinv, scores - mu))
    return m, lead

def perm_null_between(X, labels, rng, n=N_PERM):
    g = X.mean(0); total = ((X - g) ** 2).sum()
    sizes = [int(np.sum(labels == k)) for k in np.unique(labels)]
    idx = np.arange(len(X)); out = []
    for _ in range(n):
        rng.shuffle(idx); lab = np.empty(len(X), int); s = 0
        for ki, sz in enumerate(sizes):
            lab[idx[s:s + sz]] = ki; s += sz
        b = sum(np.sum(lab == ki) * ((X[lab == ki].mean(0) - g) ** 2).sum() for ki in range(len(sizes)))
        out.append(b / total)
    return float(np.mean(out)), float(np.percentile(out, 95))

def main():
    d = cc.load()
    rid = list(d["rid"])
    ex_i = rid.index(EXAMPLE) if EXAMPLE in rid else None
    variants, dom = sc.build_variants(d)
    rng = np.random.default_rng(cc.SEED)

    # baseline reference ordering
    base_m, _ = mahalanobis_full(variants["baseline_full19"]["RES"])
    base_q90 = np.percentile(base_m, 90)
    base_tail = set(np.where(base_m >= base_q90)[0])

    order = ["baseline_full19", "drop_sentence_f1_lt_0.3", "drop_report_rho_lt_0.3",
             "drop_report_rho_lt_0", "downweight_sentence_f1", "downweight_report_rho"]
    res = {"seed": cc.SEED, "n_asd": int(len(d["P"])), "example_patient": EXAMPLE,
           "baseline_between_fraction_ref": 0.148, "variants": {}}
    for name in order:
        v = variants[name]
        m, lead = mahalanobis_full(v["RES"])
        q90 = float(np.percentile(m, 90))
        tail = set(np.where(m >= q90)[0])
        # Layer 2: 2-way split variance decomposition
        lab2, _, _ = cc.gmm_fit(lead, 2)
        _, _, _, eta2 = cc.variance_partition(v["RES"], lab2)
        null_mean, null_p95 = perm_null_between(v["RES"], lab2, rng)
        r = dict(
            kind=v["kind"], n_domains=v["n_domains"],
            layer1_mahalanobis=dict(
                mean=round(float(m.mean()), 3), p90=round(q90, 3), max=round(float(m.max()), 3),
                n_top_decile=int((m >= q90).sum())),
            layer1_vs_baseline=dict(
                rank_spearman=round(float(spearmanr(m, base_m).statistic), 3),
                top_decile_overlap=len(tail & base_tail), top_decile_n=len(base_tail)),
            layer2_variance=dict(
                between_fraction=round(float(eta2), 4), within_fraction=round(float(1 - eta2), 4),
                perm_null_between_mean=round(null_mean, 4), perm_null_between_p95=round(null_p95, 4),
                mode_sizes=[int(np.sum(lab2 == k)) for k in np.unique(lab2)]),
        )
        if ex_i is not None:
            r["example_patient_layer1"] = dict(
                mahalanobis=round(float(m[ex_i]), 3),
                percentile=round(float((m <= m[ex_i]).mean() * 100), 1))
        res["variants"][name] = r
        ex = r.get("example_patient_layer1", {})
        print(f"[{name}] Maha mean={r['layer1_mahalanobis']['mean']} "
              f"rank_rho_vs_base={r['layer1_vs_baseline']['rank_spearman']} "
              f"tail_overlap={r['layer1_vs_baseline']['top_decile_overlap']}/{r['layer1_vs_baseline']['top_decile_n']} "
              f"| L2 between={r['layer2_variance']['between_fraction']} (null {r['layer2_variance']['perm_null_between_mean']}) "
              f"| ex {EXAMPLE}={ex.get('mahalanobis')} (p{ex.get('percentile')})")
    out = BASE / "fingerprint_sensitivity_results.json"
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=2)
    print("  ->", out)

if __name__ == "__main__":
    main()
