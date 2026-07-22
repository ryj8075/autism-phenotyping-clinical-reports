"""
silver_FULL vs gold report-level validation on the CURRENT canonical data.

Adapted (2026-06-27) from silver_full_vs_gold_llama.py for:
  - full silver (489 reports / 19 domains) restricted to the 26 gold-annotated reports
  - new label schema: r["labels"] is a LIST of {domain_id, domain_code, ...} per sentence
    (gold "labels" may be plain strings or dicts -- both handled)
  - 19-domain order from domain_vectors_meta_latest.json
  - canonical type-residualized ILR + ASD prototype-centroid Mahalanobis
    (matches manuscript: _common_controlled.py load()/helmert/ilr/residualize)

Outputs (suffix _full489_26gold):
  - per-report cosine (mean, median), pooled vector-level Spearman
  - per-domain Pearson/Spearman
  - silver-vs-gold Mahalanobis-rank Spearman in TWO spaces:
      (a) canonical type-residualized ILR + ASD prototype centroid (manuscript space)
      (b) simpler raw-ILR Mahalanobis (no residualization, ASD-cohort mean/cov)
"""
import json
import os
from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

BASE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

SILVER = Path(os.environ.get(
    "SILVER_LABELS_JSONL",
    REPO_ROOT / "pipeline" / "2_silver_labeling" / "data" / "silver_label" /
    "silver_labels_26reports.jsonl",
))
GOLD = Path(os.environ.get(
    "GOLD_LABELS_JSONL",
    REPO_ROOT / "pipeline" / "2_silver_labeling" / "data" / "gold_label" /
    "gold_labels_26reports.jsonl",
))
DOMAIN_VECTORS_TSV = Path(os.environ.get(
    "DOMAIN_VECTORS_TSV",
    REPO_ROOT / "pipeline" / "3_multidomain_vectors" / "domain_frequency_vector" /
    "outputs" / "domain_vectors_proportion_latest.tsv",
))
DOMAIN_META_JSON = Path(os.environ.get(
    "DOMAIN_META_JSON",
    REPO_ROOT / "pipeline" / "3_multidomain_vectors" / "domain_frequency_vector" /
    "outputs" / "domain_vectors_meta_latest.json",
))

PSEUDOCOUNT = 1e-6
SEED = 42

def helmert(D):
    V = np.zeros((D, D - 1))
    for i in range(D - 1):
        V[:i + 1, i] = 1.0 / (i + 1)
        V[i + 1, i] = -1.0
        V[:, i] *= np.sqrt((i + 1.0) / (i + 2.0))
    return V

def clr(X, eps=PSEUDOCOUNT):
    Xp = X + eps
    Xp = Xp / Xp.sum(1, keepdims=True)
    L = np.log(Xp)
    return L - L.mean(1, keepdims=True)

def ilr(X):
    return clr(X) @ helmert(X.shape[1])

def residualize_fit(M, design):
    """Return residuals AND the per-column OLS coefficients (so external vectors
    can be residualized with the SAME cohort coefficients)."""
    R = np.zeros_like(M)
    B = np.zeros((design.shape[1], M.shape[1]))
    for j in range(M.shape[1]):
        b = np.linalg.lstsq(design, M[:, j], rcond=None)[0]
        B[:, j] = b
        R[:, j] = M[:, j] - design @ b
    return R, B

def main():
    meta = json.load(open(DOMAIN_META_JSON))
    domain_list = meta["domain_columns"]
    domain_to_idx = {d: i for i, d in enumerate(domain_list)}
    D = len(domain_list)

    silver_rows = [json.loads(l) for l in open(SILVER) if l.strip()]
    gold_rows = [json.loads(l) for l in open(GOLD) if l.strip()]
    gold_reports = sorted({r["report_id"] for r in gold_rows})

    lines = []

    def p(s=""):
        print(s)
        lines.append(s)

    # ---- silver_FULL: all silver labels per report (26 gold reports) ----
    # NEW schema: r["labels"] is a list of {domain_id,...}; count EACH label.
    silver_counts = {rid: np.zeros(D) for rid in gold_reports}
    for r in silver_rows:
        rid = r["report_id"]
        if rid not in silver_counts:
            continue
        for lab in (r.get("labels") or []):
            dom = lab.get("domain_id") if isinstance(lab, dict) else lab
            if dom in domain_to_idx:
                silver_counts[rid][domain_to_idx[dom]] += 1

    # ---- gold: labels may be strings OR dicts ----
    gold_counts = {rid: np.zeros(D) for rid in gold_reports}
    for r in gold_rows:
        for lab in (r.get("labels") or []):
            dom = lab.get("domain_id") if isinstance(lab, dict) else lab
            if dom in domain_to_idx:
                gold_counts[r["report_id"]][domain_to_idx[dom]] += 1

    S = np.stack([silver_counts[r] / max(silver_counts[r].sum(), 1) for r in gold_reports])
    G = np.stack([gold_counts[r] / max(gold_counts[r].sum(), 1) for r in gold_reports])

    p(f"=== silver_FULL vs gold (full 489/19, {len(gold_reports)} gold reports) ===\n")
    p(f"domains: {D}  gold reports: {len(gold_reports)}")
    p(f"silver sentences (26 reports): {sum(1 for r in silver_rows if r['report_id'] in silver_counts)}")
    p(f"gold sentences: {len(gold_rows)}\n")

    cos_per_report = np.array([
        float(S[i] @ G[i] / max(np.linalg.norm(S[i]) * np.linalg.norm(G[i]), 1e-12))
        for i in range(len(gold_reports))])
    p(f"Per-report cosine: mean = {cos_per_report.mean():.3f}, "
      f"median = {np.median(cos_per_report):.3f}, "
      f"range = [{cos_per_report.min():.3f}, {cos_per_report.max():.3f}]\n")

    p(f"  {'domain':30s}  {'pearson':>8s}  {'spearman':>8s}  {'silver_mean':>12s}  {'gold_mean':>9s}")
    per_domain = {}
    for j, dom in enumerate(domain_list):
        s_col, g_col = S[:, j], G[:, j]
        if s_col.std() < 1e-10 or g_col.std() < 1e-10:
            r_p, r_s = float('nan'), float('nan')
        else:
            r_p = pearsonr(s_col, g_col)[0]
            r_s = spearmanr(s_col, g_col)[0]
        per_domain[dom] = dict(pearson=r_p, spearman=r_s,
                               silver_mean=float(s_col.mean()), gold_mean=float(g_col.mean()))
        p(f"  {dom:30s}  {r_p:8.3f}  {r_s:8.3f}  {s_col.mean():12.4f}  {g_col.mean():9.4f}")

    pooled_rho = spearmanr(S.flatten(), G.flatten())[0]
    p(f"\nPooled vector-level Spearman: {pooled_rho:.3f}")

    # per-group median Spearman: A = ASD-core (7), C = format (3), B = general child-psychiatric (rest)
    CORE_A = {"social_emotional_reciprocity", "nonverbal_communication", "relationship_play",
              "stereotyped_behavior", "insistence_on_sameness", "restricted_interests", "sensory_processing"}
    FORMAT_C = {"test_scores", "other_general", "recommendations"}
    grp = lambda dm: "A" if dm in CORE_A else ("C" if dm in FORMAT_C else "B")
    grp_vals = {"A": [], "B": [], "C": []}
    for dm in domain_list:
        rs = per_domain[dm]["spearman"]
        if not np.isnan(rs):
            grp_vals[grp(dm)].append(rs)
    p("\nPer-group median Spearman:")
    for g, name in [("A", "ASD-core"), ("B", "general child-psychiatric"), ("C", "format")]:
        v = grp_vals[g]
        p(f"  {g} ({name}, n={len(v)}): median = {np.median(v):.3f}")

    # ============================================================
    #  Mahalanobis-rank agreement (Para-3 key metric)
    #  Prototype space = 346 ASD reports, type-residualized ILR,
    #  prototype = cohort centroid + cov (manuscript canonical).
    # ============================================================
    df = pd.read_csv(DOMAIN_VECTORS_TSV, sep="\t")
    sub = df[df["asd_label"].values == 1].reset_index(drop=True)
    rid_asd = sub["report_id"].values
    P_asd = sub[domain_list].values.astype(float)
    rtype = np.array([r.split("-")[-1][0] for r in rid_asd])
    design_asd = np.column_stack([np.ones(len(P_asd)), (rtype == "P").astype(float)])

    ILR_asd = ilr(P_asd)
    RES_asd, B = residualize_fit(ILR_asd, design_asd)  # B = OLS coefs to reuse externally
    mu_res = RES_asd.mean(0)
    cov_inv_res = np.linalg.inv(np.cov(RES_asd, rowvar=False))

    # simpler raw-ILR (no residualization) ASD prototype
    mu_raw = ILR_asd.mean(0)
    cov_inv_raw = np.linalg.inv(np.cov(ILR_asd, rowvar=False))

    # report-type design for the 26 gold reports (for residualizing external vectors)
    rtype_gold = np.array([r.split("-")[-1][0] for r in gold_reports])
    design_gold = np.column_stack([np.ones(len(gold_reports)), (rtype_gold == "P").astype(float)])

    def mahal_residualized(V):
        Vi = ilr(V)                      # 26 x (D-1)
        Vres = Vi - design_gold @ B      # apply COHORT OLS coefs (same residualization)
        d = Vres - mu_res
        return np.sqrt(np.einsum("ij,jk,ik->i", d, cov_inv_res, d))

    def mahal_raw(V):
        Vi = ilr(V)
        d = Vi - mu_raw
        return np.sqrt(np.einsum("ij,jk,ik->i", d, cov_inv_raw, d))

    s_mahal_res = mahal_residualized(S)
    g_mahal_res = mahal_residualized(G)
    s_mahal_raw = mahal_raw(S)
    g_mahal_raw = mahal_raw(G)

    rho_res = spearmanr(s_mahal_res, g_mahal_res)[0]
    rho_raw = spearmanr(s_mahal_raw, g_mahal_raw)[0]

    p("\n=== Mahalanobis prototypicality (silver_FULL vs gold), per report ===")
    p(f"  prototype space = 346 ASD reports")
    p(f"  {'report':20s}  {'s_mahal_res':>11s}  {'g_mahal_res':>11s}  {'s_mahal_raw':>11s}  {'g_mahal_raw':>11s}")
    for i, rid in enumerate(gold_reports):
        p(f"  {rid:20s}  {s_mahal_res[i]:11.3f}  {g_mahal_res[i]:11.3f}  "
          f"{s_mahal_raw[i]:11.3f}  {g_mahal_raw[i]:11.3f}")

    p("")
    p(f"Mahalanobis-rank Spearman (type-residualized ILR, ASD prototype centroid): {rho_res:.3f}")
    p(f"Mahalanobis-rank Spearman (raw ILR, ASD prototype centroid):              {rho_raw:.3f}")
    p("")
    p("CAVEAT: external (single) gold/silver vectors are type-residualized by applying the")
    p("cohort OLS coefficients (B) fit on the 346 ASD ILR vectors to each gold report's")
    p("report-type design (A vs P). This is the consistent linear residualization; it is NOT")
    p("a re-fit on the 26 external vectors. Both silver and gold use the identical projection.")

    # per-report Mahalanobis + top-decile tail-membership agreement (for supplementary table)
    per_report_mahal = [dict(report_id=rid,
                             silver_mahal_res=round(float(s_mahal_res[i]), 3),
                             gold_mahal_res=round(float(g_mahal_res[i]), 3),
                             silver_mahal_raw=round(float(s_mahal_raw[i]), 3),
                             gold_mahal_raw=round(float(g_mahal_raw[i]), 3))
                        for i, rid in enumerate(gold_reports)]
    s_tail = s_mahal_res >= np.percentile(s_mahal_res, 90)
    g_tail = g_mahal_res >= np.percentile(g_mahal_res, 90)
    tail = dict(threshold_pct=90, n_reports=len(gold_reports),
                agreement_rate=round(float(np.mean(s_tail == g_tail)), 3),
                n_both_in_tail=int(np.sum(s_tail & g_tail)),
                n_silver_tail=int(s_tail.sum()), n_gold_tail=int(g_tail.sum()))

    # ---- write outputs ----
    txt = RESULTS / "output_full_vs_gold_llama_full489_26gold.txt"
    txt.write_text("\n".join(lines) + "\n")

    out = dict(
        n_domains=D, n_gold_reports=len(gold_reports), gold_reports=gold_reports,
        per_report_cosine=dict(mean=float(cos_per_report.mean()),
                               median=float(np.median(cos_per_report)),
                               min=float(cos_per_report.min()), max=float(cos_per_report.max())),
        pooled_vector_spearman=float(pooled_rho),
        per_domain=per_domain,
        mahalanobis_rank_spearman=dict(
            type_residualized_ilr_asd_prototype=float(rho_res),
            raw_ilr_asd_prototype=float(rho_raw)),
        per_report_mahalanobis=per_report_mahal,
        tail_membership=tail,
        prototype_space="346 ASD reports, type-residualized ILR, centroid+cov",
        caveat=("external gold/silver vectors residualized via cohort OLS coefficients applied "
                "to each gold report's A/P design; identical projection for silver and gold."),
    )
    json.dump(out, open(RESULTS / "results_full_vs_gold_llama_full489_26gold.json", "w"),
              indent=2, ensure_ascii=False)
    p(f"\n[saved] {txt}")
    p(f"[saved] {RESULTS / 'results_full_vs_gold_llama_full489_26gold.json'}")

if __name__ == "__main__":
    main()
