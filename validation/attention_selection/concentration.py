#!/usr/bin/env python3

from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import _common as C

def bh_correct(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest
    ranked_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(ranked_sorted, 0, 1)
    return q

def main() -> None:
    C.ensure_dirs()
    domain_order, codes, id_to_code, id_to_group = C.load_domains()
    store = C.AttentionStore()
    silver = C.load_sentence_labels(C.PATHS["silver_labels"])
    silver_by_report = C.report_to_sentence_indices(silver)
    silver_report_ids = sorted(silver_by_report.keys())

    K = C.TOP_K_PRIMARY
    idx = {d: i for i, d in enumerate(domain_order)}

    high_mass = np.zeros(len(domain_order))
    low_mass = np.zeros(len(domain_order))

    def add_mass(target: np.ndarray, labels: List[str]) -> None:
        active = [d for d in labels if d in idx]
        if not active:
            return
        w = 1.0 / len(active)
        for d in active:
            target[idx[d]] += w

    for rid in silver_report_ids:
        avail = set(silver_by_report[rid])
        top_idx = set(store.top_k_indices(rid, K))
        for s in avail:
            labs = silver.get((rid, s), [])
            if s in top_idx:
                add_mass(high_mass, labs)
            else:
                add_mass(low_mass, labs)

    high_total = high_mass.sum()
    low_total = low_mass.sum()

    rows = []
    pvals = []
    for d in domain_order:
        i = idx[d]
        h, l = high_mass[i], low_mass[i]
        p_high = h / high_total if high_total > 0 else 0.0
        p_low = l / low_total if low_total > 0 else 0.0
        # log-odds with Haldane-Anscombe 0.5 continuity correction
        a, b = h + 0.5, (high_total - h) + 0.5
        c, e = l + 0.5, (low_total - l) + 0.5
        log_odds = np.log((a / b) / (c / e))
        # two-proportion z-test (pooled), using mass as effective counts
        n1, n2 = high_total, low_total
        p_pool = (h + l) / (n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        z = (p_high - p_low) / se if se > 0 else 0.0
        pval = 2 * (1 - norm.cdf(abs(z)))
        pvals.append(pval)
        rows.append({
            "domain_id": d,
            "code": id_to_code[d],
            "group": id_to_group[d],
            "high_mass": float(h),
            "low_mass": float(l),
            "prop_high": float(p_high),
            "prop_low": float(p_low),
            "prop_diff": float(p_high - p_low),
            "log_odds_high_vs_low": float(log_odds),
            "z": float(z),
            "p": float(pval),
        })

    q = bh_correct(np.array(pvals))
    for r, qv in zip(rows, q):
        r["q_bh"] = float(qv)

    df2a = pd.DataFrame(rows)
    GORDER = {"CO": 0, "AS": 1, "RE": 2}
    df2a["_g"] = df2a["code"].str[:2].map(GORDER)
    df2a = df2a.sort_values(["_g", "code"]).drop(columns="_g").reset_index(drop=True)
    df2a.to_csv(C.OUTPUTS_DIR / "concentration_enrichment.csv", index=False)

    rng = np.random.default_rng(C.RANDOM_SEED)

    def silver_vec_for(rid: str, sent_indices: List[int]) -> Optional[np.ndarray]:
        labs = [silver.get((rid, s), []) for s in sent_indices]
        return C.build_vector(labs, domain_order)

    # Build matrices; drop reports with a degenerate (None) vector for that source
    y_all = np.array([store.diagnosis(rid) for rid in silver_report_ids])

    def build_matrix(source: str, rand_seed: int = 0):
        X, y = [], []
        local_rng = np.random.default_rng(rand_seed)
        for rid, yy in zip(silver_report_ids, y_all):
            avail = silver_by_report[rid]
            if source == "topK":
                idxs = store.top_k_indices(rid, K)
            elif source == "randomK":
                kk = min(K, len(avail))
                idxs = list(local_rng.choice(avail, size=kk, replace=False))
            elif source == "full":
                idxs = avail
            v = silver_vec_for(rid, idxs)
            if v is None:
                continue
            X.append(v)
            y.append(yy)
        return np.array(X), np.array(y)

    def cv_auroc(X: np.ndarray, y: np.ndarray, seed: int = C.RANDOM_SEED) -> List[float]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        aurocs = []
        for tr, te in skf.split(X, y):
            clf = Pipeline([
                ("sc", StandardScaler()),
                ("lr", LogisticRegression(max_iter=2000, C=1.0)),
            ])
            clf.fit(X[tr], y[tr])
            p = clf.predict_proba(X[te])[:, 1]
            aurocs.append(roc_auc_score(y[te], p))
        return aurocs

    auroc_results: Dict = {"config": {"K": K, "n_reports": len(silver_report_ids),
                                      "cv_folds": 5, "seed": C.RANDOM_SEED}}

    # topK and full: deterministic selection
    for source in ["topK", "full"]:
        X, y = build_matrix(source)
        a = cv_auroc(X, y)
        auroc_results[source] = {
            "auroc_mean": float(np.mean(a)),
            "auroc_sd": float(np.std(a, ddof=1)),
            "folds": [float(x) for x in a],
            "n": int(X.shape[0]),
        }

    # randomK: average over a few random selections x CV
    rand_means = []
    rand_folds_all = []
    for s in range(5):
        X, y = build_matrix("randomK", rand_seed=1000 + s)
        a = cv_auroc(X, y, seed=C.RANDOM_SEED + s)
        rand_means.append(np.mean(a))
        rand_folds_all.extend(a)
    auroc_results["randomK"] = {
        "auroc_mean": float(np.mean(rand_means)),
        "auroc_sd": float(np.std(rand_folds_all, ddof=1)),
        "n_selection_seeds": 5,
        "per_selection_mean": [float(x) for x in rand_means],
    }

    with open(C.OUTPUTS_DIR / "concentration_auroc.json", "w", encoding="utf-8") as f:
        json.dump(auroc_results, f, ensure_ascii=False, indent=2)

    _fig_enrichment(df2a)
    _fig_auroc(auroc_results)
    _print_summary(df2a, auroc_results)

def _fig_enrichment(df: pd.DataFrame) -> None:
    C.apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    GORDER = {"CO": 0, "AS": 1, "RE": 2}
    d = df.assign(_g=df["code"].str[:2].map(GORDER))
    d = d.sort_values(["_g", "code"], ascending=False).drop(columns="_g").reset_index(drop=True)
    ypos = np.arange(len(d))
    colors = [C.GROUP_COLORS[g] for g in d["group"]]

    fig, ax = plt.subplots(figsize=(6.2, 7.0))
    ax.barh(ypos, d["log_odds_high_vs_low"], color=colors, edgecolor="none")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{c}  {did}" for c, did in zip(d["code"], d["domain_id"])],
                       fontsize=8)
    ax.set_xlabel("Log-odds of label mass, high-attention vs background")

    for yi, (lo, qv) in enumerate(zip(d["log_odds_high_vs_low"], d["q_bh"])):
        if qv < 0.05:
            offset = 0.04 * (1 if lo >= 0 else -1) * max(abs(d["log_odds_high_vs_low"]).max(), 1e-6)
            ha = "left" if lo >= 0 else "right"
            ax.text(lo + offset, yi, "*", va="center", ha=ha, fontsize=11)

    legend = [
        Patch(facecolor=C.GROUP_COLORS["CO"], label="CO: Core ASD domains"),
        Patch(facecolor=C.GROUP_COLORS["AS"], label="AS: Associated and co-occurring features"),
        Patch(facecolor=C.GROUP_COLORS["RE"], label="RE: Report elements and other"),
    ]
    ax.legend(handles=legend, frameon=False, fontsize=8, loc="lower right")
    C.save_fig(fig, "concentration_enrichment_forest")
    plt.close(fig)

def _fig_auroc(res: Dict) -> None:
    C.apply_style()
    import matplotlib.pyplot as plt

    order = ["topK", "randomK", "full"]
    labels = ["Top-K attention", "Random-K", "Full report"]
    means = [res[s]["auroc_mean"] for s in order]
    sds = [res[s]["auroc_sd"] for s in order]
    colors = ["#C0392B", "#95A5A6", "#2E86C1"]

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    x = np.arange(len(order))
    ax.bar(x, means, yerr=sds, capsize=4, color=colors, edgecolor="none")
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Cross-validated AUROC (ASD vs non-ASD)")
    ax.set_ylim(0.4, 1.0)
    for xi, m in zip(x, means):
        ax.text(xi, m + 0.01, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    C.save_fig(fig, "concentration_auroc")
    plt.close(fig)

def _print_summary(df2a: pd.DataFrame, res: Dict) -> None:
    print("\n" + "=" * 70)
    print("  TEST 2a — Content enrichment (high vs background), top enriched:")
    print("=" * 70)
    top_enr = df2a.sort_values("log_odds_high_vs_low", ascending=False).head(6)
    for _, r in top_enr.iterrows():
        print("   %-3s %-28s logOR=%+.2f  q=%.3g" %
              (r["code"], r["domain_id"], r["log_odds_high_vs_low"], r["q_bh"]))
    print("  most depleted:")
    bot = df2a.sort_values("log_odds_high_vs_low").head(4)
    for _, r in bot.iterrows():
        print("   %-3s %-28s logOR=%+.2f  q=%.3g" %
              (r["code"], r["domain_id"], r["log_odds_high_vs_low"], r["q_bh"]))
    print("=" * 70)
    print("  TEST 2b — AUROC: top-K=%.3f+/-%.3f  random-K=%.3f+/-%.3f  full=%.3f+/-%.3f"
          % (res["topK"]["auroc_mean"], res["topK"]["auroc_sd"],
             res["randomK"]["auroc_mean"], res["randomK"]["auroc_sd"],
             res["full"]["auroc_mean"], res["full"]["auroc_sd"]))
    print("=" * 70)

if __name__ == "__main__":
    main()
