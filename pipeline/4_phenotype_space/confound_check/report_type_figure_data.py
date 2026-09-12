# -*- coding: utf-8 -*-
"""Report-type confound, figure data for Supplementary Figure S4.

Restricted to the 346 ASD reports, so report type is not confounded with
diagnosis, and it writes only the quantities the figure needs.

Run with the Python environment that provides numpy, pandas, scipy, and sklearn.
"""
import json, os, sys
from pathlib import Path
import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
CC_DIR = str(Path(os.environ.get(
    "PHENOTYPE_SPACE_DIR",
    REPO_ROOT / "pipeline" / "4_phenotype_space",
)))
sys.path.insert(0, CC_DIR)
import _common_controlled as cc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "report_type_figure_data.json")

def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for i in range(n - 1, -1, -1):
        idx = order[i]; prev = min(prev, p[idx] * n / (i + 1)); q[idx] = prev
    return q

def main():
    d = cc.load()
    P, rtype, dom = d["P"], d["rtype"], d["dom"]
    t = (rtype == "P").astype(float)                    # 1 = psychological assessment report
    n = len(P)

    # per-domain CLR variance explained by report type
    CLR = cc.clr(P)
    r2, pv = [], []
    for g in range(CLR.shape[1]):
        r, p = stats.pearsonr(CLR[:, g], t)             # point-biserial
        r2.append(float(r ** 2)); pv.append(float(p))
    q = bh_fdr(pv)

    # ILR variance share by report type
    ilr_r2 = [float(stats.pearsonr(d["ILR"][:, j], t)[0] ** 2) for j in range(d["ILR"].shape[1])]

    # PC1 before vs after residualization
    pc1_raw = cc.pca_scores(d["ILR"])[1][:, 0]
    pc1_res = cc.pca_scores(d["RES"])[1][:, 0]
    r_raw = float(stats.pearsonr(pc1_raw, t)[0])
    r_res = float(stats.pearsonr(pc1_res, t)[0])

    out = {
        "n_reports": int(n),
        "n_A": int((rtype == "A").sum()), "n_P": int((rtype == "P").sum()),
        "domains": list(dom),
        "panel_a": {"domain": list(dom),
                    "r2_report_type": [round(x, 4) for x in r2],
                    "fdr_q": [round(float(x), 4) for x in q],
                    "significant_q10": [bool(x <= 0.10) for x in q]},
        "panel_b": {"report_type": ["P" if x == "P" else "A" for x in rtype],
                    "pc1_raw": [round(float(x), 4) for x in pc1_raw],
                    "pc1_residual": [round(float(x), 4) for x in pc1_res],
                    "corr_pc1_type_raw": round(r_raw, 3),
                    "corr_pc1_type_residual": round(r_res, 3)},
        "ilr_variance_share_by_type": {"mean": round(float(np.mean(ilr_r2)), 4),
                                       "max": round(float(np.max(ilr_r2)), 4)},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=2)

    print(f"n = {n} ASD reports ({out['n_A']} A-type / {out['n_P']} P-type)")
    print(f"ILR variance share by report type: mean {out['ilr_variance_share_by_type']['mean']*100:.1f}% "
          f"| max {out['ilr_variance_share_by_type']['max']*100:.1f}%")
    print(f"PC1 r^2 by report type: {r_raw**2*100:.1f}%")
    print(f"domains with FDR q <= 0.10: {sum(out['panel_a']['significant_q10'])} of {len(dom)}")
    print(f"PC1 correlation with report type: {r_raw:.3f} (raw) -> {r_res:.3f} (residual)")
    print("  ->", OUT)

if __name__ == "__main__":
    main()
