# -*- coding: utf-8 -*-

import os
import sys, json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CC = str(Path(os.environ.get(
    "PHENOTYPE_SPACE_DIR",
    REPO_ROOT / "pipeline" / "4_phenotype_space",
)))
sys.path.insert(0, CC)
import _common_controlled as cc  # noqa: E402

VALID = Path(os.environ.get("SILVER_VALIDATION_DIR", REPO_ROOT / "validation" / "silver_labels"))
# Written by ../sentence_level.py (into its own folder) and ../report_level.py
# (into its results/ subfolder). Run both before any script in this folder.
PRF = Path(os.environ.get(
    "SENTENCE_LEVEL_VALIDATION_JSON",
    VALID / "sentence_level_validation_results.json",
))
REPORT_JSON = Path(os.environ.get(
    "REPORT_LEVEL_VALIDATION_JSON",
    VALID / "results" / "results_full_vs_gold_llama_full489_26gold.json",
))

def fidelity():
    """Return (f1_by_domain, report_rho_by_domain)."""
    for path, producer in ((PRF, "validation/silver_labels/sentence_level.py"),
                           (REPORT_JSON, "validation/silver_labels/report_level.py")):
        if not path.exists():
            raise FileNotFoundError(
                f"gold-fidelity input not found: {path}\n"
                f"Run {producer} first, or point the corresponding environment "
                f"variable (SENTENCE_LEVEL_VALIDATION_JSON / "
                f"REPORT_LEVEL_VALIDATION_JSON) at an existing file.")
    f1 = {r["domain"]: r["f1"] for r in json.load(open(PRF))["per_domain"]}
    rho = {d: v["spearman"] for d, v in json.load(open(REPORT_JSON))["per_domain"].items()}
    return f1, rho

def build_variants(d):
    """d = cc.load(). Returns (variants_dict, dom_list).

    Each value: dict(kind, RES, n_domains, ilr_dim, kept/dropped OR weights).
    """
    P, dom = d["P"], list(d["dom"])
    design = np.column_stack([np.ones(len(P)), (d["rtype"] == "P").astype(float)])
    f1, rho = fidelity()

    exclusions = {
        "baseline_full19": set(),
        "drop_sentence_f1_lt_0.3": {x for x in dom if f1.get(x, 1.0) < 0.3},
        "drop_report_rho_lt_0.3": {x for x in dom if rho.get(x, 1.0) < 0.3},
        "drop_report_rho_lt_0": {x for x in dom if rho.get(x, 1.0) < 0.0},
    }
    weights = {
        "downweight_sentence_f1": np.array([float(np.clip(f1.get(x, 1.0), 0, 1)) for x in dom]),
        "downweight_report_rho": np.array([max(0.0, float(rho.get(x, 0.0))) for x in dom]),
    }

    out = {}
    for name, drop in exclusions.items():
        idx = [i for i, x in enumerate(dom) if x not in drop]
        RES = cc.residualize(cc.ilr(P[:, idx]), design)
        out[name] = dict(kind="exclude", RES=RES, n_domains=len(idx), ilr_dim=len(idx) - 1,
                         kept=[dom[i] for i in idx], dropped=sorted(drop))
    for name, w in weights.items():
        Pw = P * w[None, :]
        RES = cc.residualize(cc.ilr(Pw), design)
        out[name] = dict(kind="downweight", RES=RES, n_domains=int((w > 0).sum()),
                         ilr_dim=len(dom) - 1,
                         weights={dom[i]: round(float(w[i]), 3) for i in range(len(dom))})
    return out, dom


def bestk_stability(lead, k, seed=None, n_seeds=30, n_boot=1000):
    """Reproducibility of the BIC-preferred partition, as in topk_sensitivity.cluster_stability_ari:
    cross-seed ARI over n_seeds single-init fits, and bootstrap ARI of a refit on a resample against
    the reference partition. Reported alongside the k = 2 value so the table shows both the coarsest
    split and the split BIC actually prefers."""
    import numpy as np
    from itertools import combinations
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import adjusted_rand_score
    seed = cc.SEED if seed is None else seed
    n = lead.shape[0]
    g = lambda kk, s, ni: GaussianMixture(kk, covariance_type="full", n_init=ni,
                                          max_iter=500, random_state=s)
    labs = [g(k, s, 1).fit(lead).predict(lead) for s in range(n_seeds)]
    seed_aris = [adjusted_rand_score(labs[i], labs[j]) for i, j in combinations(range(n_seeds), 2)]
    rng = np.random.default_rng(seed)
    ref = g(k, seed, 10).fit(lead).predict(lead)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = adjusted_rand_score(ref, g(k, b, 10).fit(lead[idx]).predict(lead))
    return {"k": int(k),
            "cross_seed_ari": round(float(np.mean(seed_aris)), 3),
            "bootstrap_ari": round(float(boot.mean()), 3),
            "bootstrap_ari_ci95": [round(float(np.percentile(boot, 2.5)), 3),
                                   round(float(np.percentile(boot, 97.5)), 3)]}
