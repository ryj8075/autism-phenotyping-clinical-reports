#!/usr/bin/env python3
"""
top10_report_level.py — report-level silver-vs-gold validation on the
SAME sentence set that the phenotype vectors use.

WHY
  The phenotype vectors of the main analysis are built from the ten highest-attention
  sentences of a report. The earlier report-level validation
  (report_level.py) instead pools ALL labeled sentences on both
  sides, so the object being validated is not the object being analysed. That matters most
  for Layer 1: Supplementary Table 8 reports a silver-vs-gold Mahalanobis rank correlation,
  and the distances there are computed for all-sentence profiles inside a cohort space that
  is estimated from ten-sentence vectors. This script rebuilds both sides from the ten
  attention-selected sentences, so validation and analysis use the same construction.

CONSTRUCTION (identical to the phenotype vectors)
  - ten attention-selected sentences per report (high_attention_sentences_latest.tsv)
  - 1/m mass rule: a sentence contributes one unit of mass split equally across its domains
  - L1 normalisation to a 19-domain proportion vector

SENTENCE PAIRING
  The attention table stores detokenised text (spacing around punctuation differs), so the
  ten sentences are selected by sentence_idx against the silver labels, which share the
  encoding indices. Gold rows are then paired to those silver sentences by exact sentence
  text within a report, because the sentence numbering of the gold file diverges from the
  silver file for five reports. All 26 x 10 sentences pair without loss.

METRICS (same as the all-sentence analysis, for a like-for-like comparison)
  - per-report cosine, pooled vector-level Spearman over 26 x 19 entries
  - per-domain Pearson / Spearman and per-group medians
  - Mahalanobis prototypicality rank agreement in the type-adjusted ILR space (cohort OLS
    coefficients from the 346 ASD reports) and in raw ILR, with a Fisher 95% CI

OUTPUTS (results/)
  results_top10_vs_top10.json     all numbers, incl. the all-sentence reference
  per_report_top10_vs_top10.csv   per-report cosine and Mahalanobis distances
  per_domain_top10_vs_top10.csv   per-domain agreement, top-10 and all-sentence side by side
  output_top10_vs_top10.txt       printed log
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

REPO_ROOT = BASE.parents[1]
DATA = Path(os.environ.get("SILVER_LABELING_DATA_DIR", REPO_ROOT / "pipeline" / "2_silver_labeling" / "data"))
VEC = Path(os.environ.get("DOMAIN_VECTOR_DIR",
                          REPO_ROOT / "pipeline" / "3_multidomain_vectors" / "domain_frequency_vector" / "outputs"))
SILVER = Path(os.environ.get("SILVER_LABELS_JSONL", DATA / "silver_label" / "silver_labels_489reports.jsonl"))
GOLD = Path(os.environ.get("GOLD_LABELS_JSONL", DATA / "gold_label" / "gold_labels_26reports.jsonl"))
ATTENTION_TSV = Path(os.environ.get("HIGH_ATTENTION_SENTENCES_TSV",
                                    REPO_ROOT / "pipeline" / "3_multidomain_vectors" / "top_10_sentences" /
                                    "high_attention_sentences_latest.tsv"))
DOMAIN_VECTORS_TSV = Path(os.environ.get("DOMAIN_VECTORS_TSV", VEC / "domain_vectors_proportion_latest.tsv"))
DOMAIN_META_JSON = Path(os.environ.get("DOMAIN_META_JSON", VEC / "domain_vectors_meta_latest.json"))
ALL_SENTENCE_JSON = Path(os.environ.get("REPORT_LEVEL_VALIDATION_JSON",
                                        BASE / "results" / "results_full_vs_gold_llama_full489_26gold.json"))

PSEUDOCOUNT = 1e-6
TOP_K = 10


def helmert(D: int) -> np.ndarray:
    V = np.zeros((D, D - 1))
    for i in range(D - 1):
        V[: i + 1, i] = 1.0 / (i + 1)
        V[i + 1, i] = -1.0
        V[:, i] *= np.sqrt((i + 1.0) / (i + 2.0))
    return V


def clr(X: np.ndarray, eps: float = PSEUDOCOUNT) -> np.ndarray:
    Xp = X + eps
    Xp = Xp / Xp.sum(1, keepdims=True)
    L = np.log(Xp)
    return L - L.mean(1, keepdims=True)


def ilr(X: np.ndarray) -> np.ndarray:
    return clr(X) @ helmert(X.shape[1])


def residualize_fit(M: np.ndarray, design: np.ndarray):
    R = np.zeros_like(M)
    B = np.zeros((design.shape[1], M.shape[1]))
    for j in range(M.shape[1]):
        b = np.linalg.lstsq(design, M[:, j], rcond=None)[0]
        B[:, j] = b
        R[:, j] = M[:, j] - design @ b
    return R, B


def read_jsonl(path: Path):
    return [json.loads(line) for line in open(path) if line.strip()]


def domains_of(row, domain_to_idx):
    return [d for d in ((lab.get("domain_id") if isinstance(lab, dict) else lab)
                        for lab in (row.get("labels") or [])) if d in domain_to_idx]


def mass_vector(rows, domain_to_idx, n_domains):
    """1/m mass rule over a set of sentence rows, L1-normalised."""
    v = np.zeros(n_domains)
    for row in rows:
        doms = domains_of(row, domain_to_idx)
        if not doms:
            continue
        for dom in doms:
            v[domain_to_idx[dom]] += 1.0 / len(doms)
    return v


def fisher_ci(rho: float, n: int):
    """Fisher-z interval for a SPEARMAN correlation (Bonett-Wright standard error)."""
    z = np.arctanh(rho)
    se = np.sqrt(1.06 / (n - 3))
    return [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]


def group_of(j: int) -> str:
    return "Core" if j < 7 else ("Associated" if j < 16 else "Report elements")


def main() -> None:
    lines = []

    def p(s: str = "") -> None:
        print(s)
        lines.append(s)

    domain_list = json.load(open(DOMAIN_META_JSON))["domain_columns"]
    domain_to_idx = {d: i for i, d in enumerate(domain_list)}
    D = len(domain_list)

    silver_rows = read_jsonl(SILVER)
    gold_rows = read_jsonl(GOLD)
    gold_reports = sorted({r["report_id"] for r in gold_rows})

    silver_by_key = {(r["report_id"], str(r["sentence_idx"])): r for r in silver_rows}
    gold_by_text = defaultdict(dict)
    gold_dup_text = defaultdict(set)
    for r in gold_rows:
        key = r["sentence"].strip()
        if key in gold_by_text[r["report_id"]]:
            gold_dup_text[r["report_id"]].add(key)
        gold_by_text[r["report_id"]][key] = r

    att = pd.read_csv(ATTENTION_TSV, sep="\t")
    att = att[att["report_id"].isin(gold_reports)]

    # ---- build the ten-sentence vectors on both sides ----
    S = np.zeros((len(gold_reports), D))
    G = np.zeros((len(gold_reports), D))
    coverage = {}
    for i, rid in enumerate(gold_reports):
        idxs = att.loc[att["report_id"] == rid].sort_values("rank")["sentence_idx"].tolist()[:TOP_K]
        s_rows, g_rows = [], []
        for si in idxs:
            srow = silver_by_key.get((rid, str(si)))
            if srow is None:
                raise KeyError(f"attention sentence not found in silver labels: {rid} idx {si}")
            s_rows.append(srow)
            key = srow["sentence"].strip()
            if key in gold_dup_text[rid]:
                raise ValueError(f"selected sentence occurs more than once in gold: {rid} idx {si}")
            grow = gold_by_text[rid].get(key)
            if grow is None:
                raise KeyError(f"attention sentence not found in gold labels: {rid} idx {si}")
            g_rows.append(grow)
        coverage[rid] = len(s_rows)
        sv, gv = mass_vector(s_rows, domain_to_idx, D), mass_vector(g_rows, domain_to_idx, D)
        S[i], G[i] = sv / max(sv.sum(), 1.0), gv / max(gv.sum(), 1.0)

    canon = pd.read_csv(DOMAIN_VECTORS_TSV, sep="\t").set_index("report_id")
    canon_S = canon.loc[gold_reports, domain_list].values.astype(float)
    # the canonical TSV stores proportions rounded to four decimals, so agreement is checked at
    # that precision; silver and gold are both built here at full precision, which keeps them symmetric
    max_dev = float(np.abs(S - canon_S).max())
    if max_dev > 1e-3:
        raise ValueError(f"silver ten-sentence vectors differ from the canonical phenotype vectors "
                         f"by up to {max_dev:.2e}")

    p(f"silver ten-sentence vectors reproduce the canonical phenotype vectors "
      f"(max deviation {max_dev:.1e}, TSV rounded to 4 dp)")
    p(f"=== silver vs gold on the ten attention-selected sentences, {len(gold_reports)} gold reports ===")
    if sorted(set(coverage.values())) != [TOP_K]:
        raise ValueError(f"expected {TOP_K} sentences per report, got {sorted(set(coverage.values()))}")
    p(f"sentences per report: {TOP_K} for all {len(gold_reports)} reports, paired without loss")
    p("")

    # ---- cohort prototype space: the canonical ten-sentence phenotype vectors ----
    df = pd.read_csv(DOMAIN_VECTORS_TSV, sep="\t")
    sub = df[df["asd_label"].values == 1].reset_index(drop=True)
    P_asd = sub[domain_list].values.astype(float)
    rtype_asd = np.array([r.split("-")[-1][0] for r in sub["report_id"].values])
    design_asd = np.column_stack([np.ones(len(P_asd)), (rtype_asd == "P").astype(float)])
    ILR_asd = ilr(P_asd)
    RES_asd, B = residualize_fit(ILR_asd, design_asd)
    mu_res, cov_inv_res = RES_asd.mean(0), np.linalg.inv(np.cov(RES_asd, rowvar=False))
    mu_raw, cov_inv_raw = ILR_asd.mean(0), np.linalg.inv(np.cov(ILR_asd, rowvar=False))

    rtype_gold = np.array([r.split("-")[-1][0] for r in gold_reports])
    design_gold = np.column_stack([np.ones(len(gold_reports)), (rtype_gold == "P").astype(float)])

    def mahal_adjusted(V, loo_rows=None):
        """Distance to the ASD prototype. With loo_rows, the cohort row of each scored report is
        held out before the OLS coefficients, centroid and covariance are estimated, so a report is
        never scored against a space it helped define. The silver ten-sentence vectors ARE cohort
        rows, whereas gold vectors are external, so without this the two sides are not comparable."""
        if loo_rows is None:
            d = (ilr(V) - design_gold @ B) - mu_res
            return np.sqrt(np.einsum("ij,jk,ik->i", d, cov_inv_res, d))
        out = np.empty(len(V))
        Vi = ilr(V)
        for i, row in enumerate(loo_rows):
            keep = np.ones(len(P_asd), dtype=bool)
            if row is not None:
                keep[row] = False
            RES_i, B_i = residualize_fit(ILR_asd[keep], design_asd[keep])
            mu_i = RES_i.mean(0)
            cinv_i = np.linalg.inv(np.cov(RES_i, rowvar=False))
            d = (Vi[i] - design_gold[i] @ B_i) - mu_i
            out[i] = float(np.sqrt(d @ cinv_i @ d))
        return out

    def mahal_raw(V):
        d = ilr(V) - mu_raw
        return np.sqrt(np.einsum("ij,jk,ik->i", d, cov_inv_raw, d))

    # cohort row of each gold report, if it is one of the 346 ASD reports
    cohort_pos = {r: i for i, r in enumerate(sub["report_id"].values)}
    loo_rows = [cohort_pos.get(rid) for rid in gold_reports]

    # ---- metrics ----
    num = np.einsum("ij,ij->i", S, G)
    den = np.linalg.norm(S, axis=1) * np.linalg.norm(G, axis=1)
    cos = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    pooled_sp = float(spearmanr(S.ravel(), G.ravel()).statistic)
    pooled_pe = float(pearsonr(S.ravel(), G.ravel()).statistic)

    per_dom = []
    for j, dom in enumerate(domain_list):
        constant = np.std(S[:, j]) < 1e-10 or np.std(G[:, j]) < 1e-10
        per_dom.append({"domain": dom, "group": group_of(j),
                        "spearman": np.nan if constant else float(spearmanr(S[:, j], G[:, j]).statistic),
                        "pearson": np.nan if constant else float(pearsonr(S[:, j], G[:, j]).statistic),
                        "silver_mean": float(S[:, j].mean()), "gold_mean": float(G[:, j].mean()),
                        "constant_column": bool(constant)})
    pdf = pd.DataFrame(per_dom)
    group_median = {g: float(np.nanmedian(pdf.loc[pdf.group == g, "spearman"]))
                    for g in ("Core", "Associated", "Report elements")}
    group_n_domains = {g: int(pdf.loc[pdf.group == g, "spearman"].notna().sum())
                       for g in ("Core", "Associated", "Report elements")}
    constant_domains = pdf.loc[pdf.constant_column, "domain"].tolist()

    s_adj, g_adj = mahal_adjusted(S, loo_rows), mahal_adjusted(G, loo_rows)
    s_ins, g_ins = mahal_adjusted(S), mahal_adjusted(G)      # in-sample, for sensitivity only
    s_raw, g_raw = mahal_raw(S), mahal_raw(G)
    rho_adj = float(spearmanr(s_adj, g_adj).statistic)
    rho_ins = float(spearmanr(s_ins, g_ins).statistic)
    rho_raw = float(spearmanr(s_raw, g_raw).statistic)
    n_in_cohort = int(sum(r is not None for r in loo_rows))
    n = len(gold_reports)

    out = {
        "n_gold_reports": n,
        "n_domains": D,
        "top_k": TOP_K,
        "vectors": "ten attention-selected sentences, 1/m mass rule, both silver and gold",
        "sentence_pairing": "top-10 selected by sentence_idx in silver; gold paired by exact sentence text",
        "prototype_space": "346 ASD reports, type-adjusted ILR of the ten-sentence phenotype vectors",
        "cosine": {"mean": float(np.nanmean(cos)), "median": float(np.nanmedian(cos)),
                   "min": float(np.nanmin(cos)), "max": float(np.nanmax(cos))},
        "pooled_vector_spearman": pooled_sp,
        "pooled_vector_pearson": pooled_pe,
        "per_group_median_spearman": group_median,
        "per_group_n_domains_with_variation": group_n_domains,
        "constant_domains_excluded": constant_domains,
        "per_domain": {r["domain"]: {k: r[k] for k in ("spearman", "pearson", "silver_mean", "gold_mean")}
                       for r in per_dom},
        "mahalanobis_rank_spearman": {
            "type_adjusted_leave_one_out": rho_adj, "type_adjusted_leave_one_out_ci95": fisher_ci(rho_adj, n),
            "type_adjusted_in_sample": rho_ins, "raw_ilr": rho_raw, "raw_ilr_ci95": fisher_ci(rho_raw, n),
            "n_gold_reports_inside_cohort": n_in_cohort,
            "note": ("the leave-one-out value is the reportable one: the silver ten-sentence vectors are "
                     "themselves cohort rows, so the in-sample value scores them against a space they "
                     "helped define while gold vectors stay external")},
    }

    # ---- side by side with the all-sentence analysis ----
    ref = json.load(open(ALL_SENTENCE_JSON))
    ref_mahal = ref["mahalanobis_rank_spearman"]
    ref_rho = ref_mahal["type_residualized_ilr_asd_prototype"]
    out["all_sentence_reference"] = {
        "cosine_mean": ref["per_report_cosine"]["mean"],
        "pooled_vector_spearman": ref.get("pooled_vector_spearman"),
        "mahalanobis_rank_spearman_type_adjusted": ref_rho,
        "source": str(ALL_SENTENCE_JSON),
    }

    p(f"per-report cosine: mean {out['cosine']['mean']:.3f}, median {out['cosine']['median']:.3f}, "
      f"range [{out['cosine']['min']:.3f}, {out['cosine']['max']:.3f}]")
    p(f"pooled vector Spearman: {pooled_sp:.3f}")
    p("per-group median Spearman: " + ", ".join(f"{g} {v:.3f}" for g, v in group_median.items()))
    ci = out["mahalanobis_rank_spearman"]["type_adjusted_leave_one_out_ci95"]
    p(f"Mahalanobis rank Spearman (leave-one-out): {rho_adj:.3f} "
      f"(95% CI {ci[0]:.3f} to {ci[1]:.3f}); in-sample {rho_ins:.3f}; raw ILR {rho_raw:.3f}")
    p(f"gold reports inside the 346-report cohort: {n_in_cohort} of {n}")
    p("domains with no variation (excluded from the medians): " + (", ".join(constant_domains) or "none"))
    p("domains per group median: " + ", ".join(f"{g} {v}" for g, v in group_n_domains.items()))
    p("")
    p("all-sentence reference (report_level.py):")
    a = out["all_sentence_reference"]
    p(f"  cosine mean {a['cosine_mean']:.3f}, pooled Spearman {a['pooled_vector_spearman']:.3f}, "
      f"Mahalanobis rank {a['mahalanobis_rank_spearman_type_adjusted']:.3f}")

    pd.DataFrame({"report_id": gold_reports, "report_type": rtype_gold, "cosine": cos,
                  "silver_mahalanobis_adj_loo": s_adj, "gold_mahalanobis_adj_loo": g_adj,
                  "silver_mahalanobis_adj_insample": s_ins, "gold_mahalanobis_adj_insample": g_ins,
                  "silver_mahalanobis_raw": s_raw, "gold_mahalanobis_raw": g_raw,
                  "n_sentences": [coverage[r] for r in gold_reports]}
                 ).to_csv(RESULTS / "per_report_top10_vs_top10.csv", index=False)

    ref_dom = {d: v["spearman"] for d, v in ref["per_domain"].items()}
    pdf["spearman_all_sentences"] = pdf["domain"].map(ref_dom)
    pdf.to_csv(RESULTS / "per_domain_top10_vs_top10.csv", index=False)

    # per-report values inside the JSON, in the shape TableS8.R reads (one record per report)
    out["per_report"] = [
        {"report_id": rid, "report_type": str(rt), "cosine": float(c),
         "silver_mahalanobis_adj_loo": float(sa), "gold_mahalanobis_adj_loo": float(ga),
         "silver_mahalanobis_adj_insample": float(si), "gold_mahalanobis_adj_insample": float(gi),
         "silver_mahalanobis_raw": float(sr), "gold_mahalanobis_raw": float(gr)}
        for rid, rt, c, sa, ga, si, gi, sr, gr in zip(gold_reports, rtype_gold, cos, s_adj, g_adj,
                                                      s_ins, g_ins, s_raw, g_raw)]

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, float) and np.isnan(o):
            return None
        return o

    json.dump(clean(out), open(RESULTS / "results_top10_vs_top10.json", "w"),
              ensure_ascii=False, indent=2)
    p("")
    p(f"[saved] {RESULTS}")
    open(RESULTS / "output_top10_vs_top10.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
