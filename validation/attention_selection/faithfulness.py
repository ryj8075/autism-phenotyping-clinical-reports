#!/usr/bin/env python3
"""
test1_faithfulness.py — Does the top-K attention subset reconstruct a report's
phenotype better than (or as well as) a random K-sentence subset?

For each report we build three SILVER domain vectors:
  (a) top-K attention sentences
  (b) random-K sentences (averaged over N=200 draws, fixed seed)
  (c) FULL = all sentences of the report

References:
  PRIMARY   = expert GOLD report-level vector (26 gold reports).
              metric = cosine + Spearman of each silver vector vs gold vector.
  SECONDARY = the report's own FULL silver vector (all 489 reports).
              metric = cosine(top-K, full) vs cosine(random-K, full).

K primary = 10; sweep K in {5,10,15,20}.
Stats: paired Wilcoxon signed-rank (top-K vs random-K) per K, medians, effect size.

Outputs (outputs/):
  test1_results.json          full numeric results
  test1_per_report.csv        per-report cosine/spearman by source (K=10)
  test1_sweep.csv             summary by K
Figures (figures/):
  test1_gold_reconstruction   cosine vs gold, top-K/random/full, by K
  test1_full_reconstruction   cosine vs full silver, top-K vs random, by K
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cos_dist
from scipy.stats import spearmanr, wilcoxon

import _common as C

def cosine_sim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return np.nan
    if np.all(a == 0) or np.all(b == 0):
        return np.nan
    return float(1.0 - cos_dist(a, b))

def spearman_sim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return np.nan
    if np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan
    r, _ = spearmanr(a, b)
    return float(r)

def rank_biserial(diffs: np.ndarray) -> float:
    """Matched-pairs rank-biserial effect size for Wilcoxon (top-K vs random-K)."""
    d = diffs[diffs != 0]
    if d.size == 0:
        return 0.0
    ranks = pd.Series(np.abs(d)).rank().values
    r_pos = ranks[d > 0].sum()
    r_neg = ranks[d < 0].sum()
    total = r_pos + r_neg
    return float((r_pos - r_neg) / total) if total > 0 else 0.0

def main() -> None:
    C.ensure_dirs()
    domain_order, codes, id_to_code, id_to_group = C.load_domains()
    store = C.AttentionStore()

    silver = C.load_sentence_labels(C.PATHS["silver_labels"])
    gold = C.load_sentence_labels(C.PATHS["gold_labels"])

    silver_by_report = C.report_to_sentence_indices(silver)
    gold_by_report = C.report_to_sentence_indices(gold)

    silver_report_ids = sorted(silver_by_report.keys())
    gold_report_ids = sorted(gold_by_report.keys())

    rng = np.random.default_rng(C.RANDOM_SEED)

    def silver_vec_for(report_id: str, sent_indices: List[int]) -> Optional[np.ndarray]:
        labs = [silver.get((report_id, s), []) for s in sent_indices]
        return C.build_vector(labs, domain_order)

    # GOLD report-level vectors (built with the SAME build_vector from gold labels)
    gold_vectors: Dict[str, Optional[np.ndarray]] = {}
    for rid in gold_report_ids:
        labs = [gold[(rid, s)] for s in gold_by_report[rid]]
        gold_vectors[rid] = C.build_vector(labs, domain_order)

    # FULL silver vectors (all 489)
    full_silver_vectors: Dict[str, Optional[np.ndarray]] = {}
    for rid in silver_report_ids:
        full_silver_vectors[rid] = silver_vec_for(rid, silver_by_report[rid])

    results: Dict = {
        "config": {
            "k_primary": C.TOP_K_PRIMARY,
            "k_sweep": C.K_SWEEP,
            "n_random_draws": C.N_RANDOM_DRAWS,
            "random_seed": C.RANDOM_SEED,
            "n_silver_reports": len(silver_report_ids),
            "n_gold_reports": len(gold_report_ids),
            "n_domains": len(domain_order),
        },
        "gold_reference": {},
        "full_reference": {},
    }

    sweep_rows: List[Dict] = []
    per_report_k10: List[Dict] = []

    for K in C.K_SWEEP:
        # ---- GOLD reference (26 reports) ----
        g_cos_top, g_cos_rand, g_cos_full = [], [], []
        g_sp_top, g_sp_rand, g_sp_full = [], [], []

        for rid in gold_report_ids:
            gvec = gold_vectors[rid]
            if gvec is None:
                continue
            avail = silver_by_report[rid]

            top_idx = store.top_k_indices(rid, K)
            v_top = silver_vec_for(rid, top_idx)
            v_full = full_silver_vectors[rid]

            # random-K averaged over draws (cosine/spearman averaged)
            kk = min(K, len(avail))
            rc, rs = [], []
            for _ in range(C.N_RANDOM_DRAWS):
                pick = rng.choice(avail, size=kk, replace=False)
                v_rand = silver_vec_for(rid, list(pick))
                rc.append(cosine_sim(v_rand, gvec))
                rs.append(spearman_sim(v_rand, gvec))

            g_cos_top.append(cosine_sim(v_top, gvec))
            g_cos_rand.append(np.nanmean(rc))
            g_cos_full.append(cosine_sim(v_full, gvec))
            g_sp_top.append(spearman_sim(v_top, gvec))
            g_sp_rand.append(np.nanmean(rs))
            g_sp_full.append(spearman_sim(v_full, gvec))

            if K == C.TOP_K_PRIMARY:
                per_report_k10.append({
                    "report_id": rid,
                    "reference": "gold",
                    "cos_topK": cosine_sim(v_top, gvec),
                    "cos_randK": float(np.nanmean(rc)),
                    "cos_full": cosine_sim(v_full, gvec),
                    "spearman_topK": spearman_sim(v_top, gvec),
                    "spearman_randK": float(np.nanmean(rs)),
                    "spearman_full": spearman_sim(v_full, gvec),
                })

        g_cos_top = np.array(g_cos_top); g_cos_rand = np.array(g_cos_rand); g_cos_full = np.array(g_cos_full)
        g_sp_top = np.array(g_sp_top); g_sp_rand = np.array(g_sp_rand); g_sp_full = np.array(g_sp_full)

        def paired_test(a, b):
            m = ~(np.isnan(a) | np.isnan(b))
            a2, b2 = a[m], b[m]
            diffs = a2 - b2
            if np.all(diffs == 0) or diffs.size < 1:
                return {"wilcoxon_p": np.nan, "effect_rank_biserial": 0.0, "n": int(diffs.size)}
            try:
                stat, p = wilcoxon(a2, b2)
            except ValueError:
                p = np.nan
            return {"wilcoxon_p": float(p), "effect_rank_biserial": rank_biserial(diffs), "n": int(diffs.size)}

        gold_block = {
            "cosine": {
                "topK_median": float(np.nanmedian(g_cos_top)),
                "randK_median": float(np.nanmedian(g_cos_rand)),
                "full_median": float(np.nanmedian(g_cos_full)),
                "topK_mean": float(np.nanmean(g_cos_top)),
                "randK_mean": float(np.nanmean(g_cos_rand)),
                "full_mean": float(np.nanmean(g_cos_full)),
                "topK_vs_randK": paired_test(g_cos_top, g_cos_rand),
                "topK_vs_full": paired_test(g_cos_top, g_cos_full),
            },
            "spearman": {
                "topK_median": float(np.nanmedian(g_sp_top)),
                "randK_median": float(np.nanmedian(g_sp_rand)),
                "full_median": float(np.nanmedian(g_sp_full)),
                "topK_mean": float(np.nanmean(g_sp_top)),
                "randK_mean": float(np.nanmean(g_sp_rand)),
                "full_mean": float(np.nanmean(g_sp_full)),
                "topK_vs_randK": paired_test(g_sp_top, g_sp_rand),
            },
        }
        results["gold_reference"][str(K)] = gold_block

        # ---- FULL silver reference (all 489) ----
        f_cos_top, f_cos_rand = [], []
        f_sp_top, f_sp_rand = [], []
        for rid in silver_report_ids:
            v_full = full_silver_vectors[rid]
            if v_full is None:
                continue
            avail = silver_by_report[rid]
            top_idx = store.top_k_indices(rid, K)
            v_top = silver_vec_for(rid, top_idx)
            kk = min(K, len(avail))
            rc, rs = [], []
            for _ in range(C.N_RANDOM_DRAWS):
                pick = rng.choice(avail, size=kk, replace=False)
                v_rand = silver_vec_for(rid, list(pick))
                rc.append(cosine_sim(v_rand, v_full))
                rs.append(spearman_sim(v_rand, v_full))
            f_cos_top.append(cosine_sim(v_top, v_full))
            f_cos_rand.append(np.nanmean(rc))
            f_sp_top.append(spearman_sim(v_top, v_full))
            f_sp_rand.append(np.nanmean(rs))

        f_cos_top = np.array(f_cos_top); f_cos_rand = np.array(f_cos_rand)
        f_sp_top = np.array(f_sp_top); f_sp_rand = np.array(f_sp_rand)

        full_block = {
            "cosine": {
                "topK_median": float(np.nanmedian(f_cos_top)),
                "randK_median": float(np.nanmedian(f_cos_rand)),
                "topK_mean": float(np.nanmean(f_cos_top)),
                "randK_mean": float(np.nanmean(f_cos_rand)),
                "topK_vs_randK": paired_test(f_cos_top, f_cos_rand),
            },
            "spearman": {
                "topK_median": float(np.nanmedian(f_sp_top)),
                "randK_median": float(np.nanmedian(f_sp_rand)),
                "topK_mean": float(np.nanmean(f_sp_top)),
                "randK_mean": float(np.nanmean(f_sp_rand)),
                "topK_vs_randK": paired_test(f_sp_top, f_sp_rand),
            },
        }
        results["full_reference"][str(K)] = full_block

        sweep_rows.append({
            "K": K,
            "gold_cos_topK": gold_block["cosine"]["topK_median"],
            "gold_cos_randK": gold_block["cosine"]["randK_median"],
            "gold_cos_full": gold_block["cosine"]["full_median"],
            "gold_cos_p_top_vs_rand": gold_block["cosine"]["topK_vs_randK"]["wilcoxon_p"],
            "gold_sp_topK": gold_block["spearman"]["topK_median"],
            "gold_sp_randK": gold_block["spearman"]["randK_median"],
            "full_cos_topK": full_block["cosine"]["topK_median"],
            "full_cos_randK": full_block["cosine"]["randK_median"],
            "full_cos_p_top_vs_rand": full_block["cosine"]["topK_vs_randK"]["wilcoxon_p"],
        })

    # ---- Save numeric ----
    with open(C.OUTPUTS_DIR / "test1_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    pd.DataFrame(per_report_k10).to_csv(C.OUTPUTS_DIR / "test1_per_report.csv", index=False)
    pd.DataFrame(sweep_rows).to_csv(C.OUTPUTS_DIR / "test1_sweep.csv", index=False)

    # ---- Figures ----
    _figures(results)
    _print_summary(results)

def _figures(results: Dict) -> None:
    C.apply_style()
    import matplotlib.pyplot as plt

    Ks = C.K_SWEEP

    # Fig 1: cosine vs GOLD, top-K / random-K / full
    top = [results["gold_reference"][str(k)]["cosine"]["topK_median"] for k in Ks]
    rnd = [results["gold_reference"][str(k)]["cosine"]["randK_median"] for k in Ks]
    full = [results["gold_reference"][str(k)]["cosine"]["full_median"] for k in Ks]

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    x = np.arange(len(Ks))
    w = 0.27
    ax.bar(x - w, top, w, color="#C0392B", label="Top-K attention")
    ax.bar(x, rnd, w, color="#95A5A6", label="Random-K")
    ax.bar(x + w, full, w, color="#2E86C1", label="Full report")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in Ks])
    ax.set_xlabel("K (number of sentences)")
    ax.set_ylabel("Cosine similarity to expert gold vector (median)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    C.save_fig(fig, "test1_gold_reconstruction")
    plt.close(fig)

    # Fig 2: cosine vs FULL silver, top-K vs random-K
    top = [results["full_reference"][str(k)]["cosine"]["topK_median"] for k in Ks]
    rnd = [results["full_reference"][str(k)]["cosine"]["randK_median"] for k in Ks]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    x = np.arange(len(Ks))
    w = 0.35
    ax.bar(x - w / 2, top, w, color="#C0392B", label="Top-K attention")
    ax.bar(x + w / 2, rnd, w, color="#95A5A6", label="Random-K")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in Ks])
    ax.set_xlabel("K (number of sentences)")
    ax.set_ylabel("Cosine similarity to full-report silver vector (median)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    C.save_fig(fig, "test1_full_reconstruction")
    plt.close(fig)

def _print_summary(results: Dict) -> None:
    k = str(C.TOP_K_PRIMARY)
    g = results["gold_reference"][k]
    f = results["full_reference"][k]
    print("\n" + "=" * 70)
    print(f"  TEST 1 — Faithfulness (K={C.TOP_K_PRIMARY})")
    print("=" * 70)
    print("  vs GOLD (cosine median):  top-K=%.3f  rand-K=%.3f  full=%.3f  (p=%.3g)"
          % (g["cosine"]["topK_median"], g["cosine"]["randK_median"],
             g["cosine"]["full_median"], g["cosine"]["topK_vs_randK"]["wilcoxon_p"]))
    print("  vs GOLD (spearman med):   top-K=%.3f  rand-K=%.3f  full=%.3f  (p=%.3g)"
          % (g["spearman"]["topK_median"], g["spearman"]["randK_median"],
             g["spearman"]["full_median"], g["spearman"]["topK_vs_randK"]["wilcoxon_p"]))
    print("  vs FULL silver (cos med): top-K=%.3f  rand-K=%.3f  (p=%.3g)"
          % (f["cosine"]["topK_median"], f["cosine"]["randK_median"],
             f["cosine"]["topK_vs_randK"]["wilcoxon_p"]))
    print("=" * 70)

if __name__ == "__main__":
    main()
