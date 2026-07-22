# -*- coding: utf-8 -*-
"""Shared variant builder for the Figure 4/5/6 domain-fidelity sensitivity (steps 1-3).

Builds a common set of 6 phenotype-space representations from the SAME canonical ASD-346
type-residual pipeline (`_common_controlled`), differing only in how low-fidelity domains are
handled:
  - baseline_full19            : all 19 domains
  - drop_sentence_f1_lt_0.3    : exclude domains with sentence-level F1 < 0.3   (from analysis 2_1)
  - drop_report_rho_lt_0.3     : exclude domains with report-level Spearman < 0.3
  - drop_report_rho_lt_0       : exclude domains with report-level Spearman < 0
  - downweight_sentence_f1     : multiply each domain's proportion mass by clip(F1, 0, 1)
  - downweight_report_rho      : multiply each domain's proportion mass by max(0, rho)

Exclusion/weighting is applied on the proportion (composition) vector BEFORE clr/ILR. Weights are
fidelity scores that naturally live in [0,1] (clipped F1, or max(0, rho)); we apply them at their
natural scale, which is the reported design choice. NOTE: clr (`_common_controlled.clr`) is NOT
exactly scale-invariant here because it adds a FIXED additive PSEUDOCOUNT (1e-6) before
renormalizing -- so the *relative* weights are what chiefly drive the result, but the absolute
weight scale is not fully irrelevant: rescaling the whole weight vector shifts how the pseudocount
floors near-zero-weight domains. A domain weighted to ~0 (e.g. physiological_function, F1=0) is
therefore effectively soft-dropped with a pseudocount floor, i.e. down-weighting shades into
exclusion at the low end. Each variant returns its type-residualized ILR matrix RES, computed with
the identical design (intercept + report-type P) used by the canonical pipeline, so
baseline_full19.RES == cc.load()["RES"] exactly.
"""
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
PRF = Path(os.environ.get(
    "SENTENCE_LEVEL_VALIDATION_JSON",
    VALID / "sentence_level" / "sentence_level_validation_results.json",
))
REPORT_JSON = Path(os.environ.get(
    "REPORT_LEVEL_VALIDATION_JSON",
    VALID / "report_level" / "results" / "results_full_vs_gold_llama_full489_26gold.json",
))

def fidelity():
    """Return (f1_by_domain, report_rho_by_domain)."""
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
