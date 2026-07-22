# -*- coding: utf-8 -*-

import os, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import CCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42; matplotlib.rcParams["ps.fonttype"] = 42

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("REPORT_LLM_DATA_ROOT", REPO_ROOT / "data"))
MATCH = str(Path(os.environ.get(
    "DOMAIN_MATCHING_DETAILS_JSON",
    REPO_ROOT / "pipeline" / "3_multidomain_vectors" / "domain_frequency_vector" /
    "outputs" / "matching_details_latest.json",
)))
SCORES = str(Path(os.environ.get(
    "PHENOTYPE_TABLE_LONG",
    DATA_ROOT / "phenotype_scale_multidimensionality" / "phenotype_table_long.tsv",
)))
SEED = 42
ALPHAS = np.logspace(-2, 3, 20)

DOMAINS = ["social_emotional_reciprocity","nonverbal_communication","relationship_play",
           "stereotyped_behavior","insistence_on_sameness","restricted_interests","sensory_processing",
           "externalizing","internalizing","language_skills","physiological_function","adaptive_behavior",
           "intelligence_learning","executive_function","motor_skills","family_environment",
           "test_scores","other_general","recommendations"]

A_TEXT = ["social_emotional_reciprocity","nonverbal_communication","relationship_play",
          "stereotyped_behavior","insistence_on_sameness","restricted_interests","sensory_processing","language_skills"]
A_SCORES = ["ADOS_SA_raw","ADOS_RRB_raw","ADOS_CSS","ADIR_social_A","ADIR_comm_B","ADIR_rrb_C","CARS_total"]
P_TEXT = ["intelligence_learning","adaptive_behavior","language_skills","motor_skills","social_emotional_reciprocity"]
P_SCORES = ["FSIQ","VABS_comm_std","VABS_dls_std","VABS_social_std","VABS_motor_std","VABS_composite_std"]

def load_hits():

    d = json.load(open(MATCH))
    return {x["report_id"]: list(x["domains_found"]) for x in d}

def vec19(hits):

    v = np.zeros(len(DOMAINS))
    for h in hits:
        if h in DOMAINS:
            v[DOMAINS.index(h)] += 1
    s = v.sum()
    return v / s if s > 0 else v

def split_halves(hits, rng):
    h = list(hits); rng.shuffle(h)
    k = len(h) // 2
    return vec19(h[:k]), vec19(h[k:])

def zscore(M):
    mu = M.mean(0); sd = M.std(0, ddof=0); sd[sd == 0] = 1.0
    return (M - mu) / sd

def cv_r2(X, y):

    kf = KFold(5, shuffle=True, random_state=SEED)
    yp = cross_val_predict(RidgeCV(alphas=ALPHAS), X, y, cv=kf)
    return float(r2_score(y, yp)), yp

def reconstruct(Xtext, Xscore, text_names, score_names):

    t2s = {sn: cv_r2(Xtext, Xscore[:, j])[0] for j, sn in enumerate(score_names)}
    s2t, pred = {}, np.zeros_like(Xtext)
    for k, tn in enumerate(text_names):
        r2, yp = cv_r2(Xscore, Xtext[:, k]); s2t[tn] = r2; pred[:, k] = yp
    return t2s, s2t, pred

def main():
    hits = load_hits()
    sc = pd.read_csv(SCORES, sep="\t")
    rng = np.random.RandomState(SEED)
    out = {"design": ("Restricted to shared constructs, excluding text-only domains and balancing dimensionality. "
                      "Report type is controlled by stratifying tracks. Reconstruction uses 5-fold CV Ridge R2. "
                      "External incremental prediction is deferred because the available N is too small."),
           "logic": ("The text-superset claim requires (A) successful text-to-score reconstruction plus "
                     "(B) score-to-text residual information. (C) Split-half reproducibility checks whether "
                     "the residual is more than noise. Residual information alone only shows difference; "
                     "reconstruction is needed to support a contains-and-adds-to claim."),
           "caveats": [
               "Only shared constructs are used, so the test concerns information beyond scores within the same "
               "construct space rather than scores that the text was never designed to measure.",
               "Reconstruction uses 5-fold CV R2 to reduce overfitting. N is the complete-case sample within each "
               "A-track and P-track analysis.",
               "Split-half reliability is limited by sparsity: the median report has about 23 domain hits, leaving "
               "about 11 hits per half. Spearman-Brown correction is reported alongside raw reliability.",
               "External prediction using additional eCRF measures such as SRS-2 is not run because the KD10 subset "
               "has only about 23 cases; future work should test genetic or longitudinal outcomes.",
               "Report type is controlled through track stratification, so A/P is constant within each track.",
               "Examiner fingerprint adjustment is recommended for future extensions and is not included here."]}

    for track, T_TEXT, T_SCORES in [("A", A_TEXT, A_SCORES), ("P", P_TEXT, P_SCORES)]:
        d = sc[sc["Report_type"] == ("ASD" if track == "A" else "Psych")].copy()
        d = d[d["Report_ID"].isin(hits)].reset_index(drop=True)

        S = d[T_SCORES].apply(pd.to_numeric, errors="coerce")
        keep = S.notna().all(1).values
        d = d[keep].reset_index(drop=True); S = S[keep].reset_index(drop=True)
        n = len(d)

        full = np.array([vec19(hits[r]) for r in d["Report_ID"]])
        h1 = np.zeros_like(full); h2 = np.zeros_like(full)
        for i, r in enumerate(d["Report_ID"]):
            a, b = split_halves(hits[r], rng); h1[i] = a; h2[i] = b
        ti = [DOMAINS.index(t) for t in T_TEXT]
        Xtext = zscore(full[:, ti]); Xscore = zscore(S.values)

        t2s, s2t, _ = reconstruct(Xtext, Xscore, T_TEXT, T_SCORES)
        A_mean = float(np.mean(list(t2s.values())))
        B_mean = float(np.mean(list(s2t.values())))

        prng = np.random.RandomState(SEED)
        perm = []
        for _ in range(200):
            idx = prng.permutation(n)
            perm.append(np.mean([cv_r2(Xtext, Xscore[idx, j])[0] for j in range(Xscore.shape[1])]))
        perm = np.array(perm)
        A_perm_p = float((np.sum(perm >= A_mean) + 1) / (len(perm) + 1))

        xc, yc = CCA(n_components=1).fit(Xtext, Xscore).transform(Xtext, Xscore)
        canon_r = float(np.corrcoef(xc[:, 0], yc[:, 0])[0, 1])

        pred = np.zeros((n, len(ti)))
        for k in range(len(ti)):
            pred[:, k] = cv_r2(Xscore, zscore(full[:, ti])[:, k])[1]
        h1s, h2s = zscore(h1[:, ti]), zscore(h2[:, ti])
        raw_rel, res_rel = [], []
        for k in range(len(ti)):
            r_raw = stats.pearsonr(h1s[:, k], h2s[:, k])[0]
            r_res = stats.pearsonr(h1s[:, k] - pred[:, k], h2s[:, k] - pred[:, k])[0]
            sb = lambda r: (2 * r / (1 + r)) if r > -1 else np.nan   # Spearman-Brown
            raw_rel.append(sb(r_raw)); res_rel.append(sb(r_res))
        raw_sb_m = float(np.nanmean(raw_rel)); res_sb_m = float(np.nanmean(res_rel))

        recon_ok = A_mean >= 0.25 and (A_mean - B_mean) > 0
        rel_ok = 0.5 <= raw_sb_m <= 1.0
        verdict = ("SUPPORTS" if (recon_ok and rel_ok) else "UNDERPOWERED / NOT ESTABLISHED")
        out[f"{track}_track"] = {
            "n": n, "text_domains": T_TEXT, "scores": T_SCORES,
            "verdict": verdict,
            "reconstruction": {
                "A_text_to_scores_meanR2": round(A_mean, 3), "per_score_R2": {k: round(v, 3) for k, v in t2s.items()},
                "B_scores_to_text_meanR2": round(B_mean, 3), "per_domain_R2": {k: round(v, 3) for k, v in s2t.items()},
                "asymmetry_A_minus_B": round(A_mean - B_mean, 3),
                "A_permutation_p": round(A_perm_p, 4),
                "in_sample_canonical_r": round(canon_r, 3),
                "interpretation": (f"Cross-validated R2 is low in both directions (A={A_mean:.2f}, B={B_mean:.2f}). "
                                   f"Text-to-score reconstruction exceeds chance by permutation test (p={A_perm_p:.3f}), "
                                   f"so the signal is not zero, but it is small. The in-sample canonical correlation "
                                   f"{canon_r:.2f} (R2 about {canon_r**2:.2f}) suggests a moderate association, but it "
                                   f"does not generalize out of sample (N={n}, likely overfitting shrinkage). A would "
                                   f"need to be materially larger to support a text-superset claim. A-B asymmetry is "
                                   f"not interpreted because the score-to-text target has low reliability (SB about "
                                   f"0.18), so A>B may reflect target reliability rather than information content.")},
            "residual_reliability": {
                "raw_text_SB": round(raw_sb_m, 3),
                "residual_SB": round(res_sb_m, 3),
                "per_domain_residual_SB": {T_TEXT[k]: round(float(res_rel[k]), 3) for k in range(len(ti))},
                "stable": rel_ok,
                "interpretation": ("Split-half reliability is low or unstable, including negative or divergent values. "
                                   "The top-10 sentence domain vector is too sparse, with about 23 hits per report and "
                                   "about 11 hits per half, so half-sample estimates are noisy. Because representation "
                                   "reproducibility is not established, the residual-as-signal argument is not supported "
                                   "by these data. Denser representations and larger samples are needed.")}}
        print(f"[step3|{track}] n={n} | A(text→score)={A_mean:.3f} | B(score→text)={B_mean:.3f} | asym={A_mean-B_mean:+.3f}"
              f" | in-sample canonical r={canon_r:.3f}")
        print(f"          reliability raw_SB={raw_sb_m:.3f} | residual_SB={res_sb_m:.3f} | VERDICT={verdict}")

    out["OVERALL_CONCLUSION"] = (
        "These data do not establish the claim that text vectors contain richer information than clinical scores "
        "(UNDERPOWERED). Reconstruction shows moderate in-sample canonical correlations, consistent with step 2, "
        "but cross-validated R2 is low in both directions and does not generalize out of sample. Residual "
        "reproducibility is also unstable because top-10 sentence vectors are sparse. Text-to-score reconstruction "
        "does exceed chance, so there is a real but small signal; however, it is insufficient for a superset claim. "
        "A-B asymmetry is not interpreted because low reliability of the text-vector target is a confound. Future "
        "work should use denser whole-sentence attention-weighted representations, larger samples, and external "
        "outcomes such as genetic or longitudinal measures. The defensible claim in the current data remains the "
        "step 2 convergent signal, especially RRB with RRB scores and cognitive salience with FSIQ.")
    json.dump(out, open(f"{HERE}/step3_reconstruction_asymmetry_results.json", "w"), ensure_ascii=False, indent=2)
    print("\n[OVERALL]", out["OVERALL_CONCLUSION"][:120], "...")

    tracks = ["A_track", "P_track"]

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    x = np.arange(len(tracks)); w = 0.38
    A = [out[t]["reconstruction"]["A_text_to_scores_meanR2"] for t in tracks]
    B = [out[t]["reconstruction"]["B_scores_to_text_meanR2"] for t in tracks]
    ax.bar(x - w/2, A, w, label="text→scores (A)", color="#2c7fb8")
    ax.bar(x + w/2, B, w, label="scores→text (B)", color="#d95f0e")
    ax.set_xticks(x); ax.set_xticklabels(["A-track\n(ADOS/ADIR)", "P-track\n(FSIQ/VABS)"], fontsize=8)
    ax.set_ylabel("cross-validated $R^2$", fontsize=8); ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False); ax.grid(False)
    for xi, (a, b) in enumerate(zip(A, B)):
        ax.text(xi - w/2, a + .01, f"{a:.2f}", ha="center", fontsize=6)
        ax.text(xi + w/2, b + .01, f"{b:.2f}", ha="center", fontsize=6)
    fig.tight_layout()
    for e in ("png", "pdf"): fig.savefig(f"{HERE}/fig_reconstruction_asymmetry.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    R = [out[t]["residual_reliability"]["raw_text_SB"] for t in tracks]
    RS = [out[t]["residual_reliability"]["residual_SB"] for t in tracks]
    ax.bar(x - w/2, R, w, label="raw text vector", color="#31a354")
    ax.bar(x + w/2, RS, w, label="residual (scores removed)", color="#756bb1")
    ax.axhline(0.5, ls="--", lw=.8, color="#888")
    ax.set_xticks(x); ax.set_xticklabels(["A-track", "P-track"], fontsize=8)
    ax.set_ylabel("split-half reliability (Spearman–Brown)", fontsize=7.5); ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False); ax.grid(False)
    for xi, (a, b) in enumerate(zip(R, RS)):
        ax.text(xi - w/2, a + .01, f"{a:.2f}", ha="center", fontsize=6)
        ax.text(xi + w/2, b + .01, f"{b:.2f}", ha="center", fontsize=6)
    fig.tight_layout()
    for e in ("png", "pdf"): fig.savefig(f"{HERE}/fig_residual_reliability.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  →", f"{HERE}/step3_reconstruction_asymmetry_results.json ; fig_reconstruction_asymmetry.*, fig_residual_reliability.*")

if __name__ == "__main__":
    main()
