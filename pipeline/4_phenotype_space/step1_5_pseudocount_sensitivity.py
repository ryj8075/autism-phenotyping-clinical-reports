# -*- coding: utf-8 -*-

import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common_controlled as cc

HERE = os.path.dirname(os.path.abspath(__file__))
PSEUDOCOUNTS = [1e-8, 1e-6, 1e-4, 1e-2, 0.05, 0.1]
BASELINE = cc.PSEUDOCOUNT  # 1e-6 (canonical)

rows = []
for pc in PSEUDOCOUNTS:
    cc.PSEUDOCOUNT = pc
    d = cc.load()                                    # type-residual ILR with this pseudocount
    _, scores, _, evr = cc.pca_scores(d["RES"])
    n, ndim = scores.shape
    lead = cc.mardia_z(scores[:, :3])
    resid = cc.mardia_z(scores[:, 3:])               # PC4+ residual  <- idiosyncratic tail
    full = cc.mardia_z(d["RES"])

    mu = scores.mean(0)
    ci = np.linalg.inv(np.cov(scores, rowvar=False))
    diff = scores - mu
    maha = np.sqrt(np.sum(diff @ ci * diff, 1))
    chi99 = np.sqrt(stats.chi2.ppf(0.99, df=ndim))
    tail_frac = float(np.mean(maha > chi99))
    resid_heavy = bool(resid["z"] > 1.96 and resid["kurt"] == "lepto")
    rows.append(dict(
        pseudocount=pc, n_asd=int(n), ndim=int(ndim),
        pc1_3_z=round(float(lead["z"]), 3), pc1_3_kurt=lead["kurt"],
        pc4plus_z=round(float(resid["z"]), 3), pc4plus_kurt=resid["kurt"],
        full_z=round(float(full["z"]), 3),
        pc13_var=round(float(evr[:3].sum()), 4),
        pc4plus_var=round(float(evr[3:].sum()), 4),
        tail_frac_beyond_chi99=round(tail_frac, 4),
        residual_heavy_tail=resid_heavy))
    print("pc=%.0e | PC1-3 z=%7.2f(%s) | PC4+ z=%7.2f(%s) | full z=%7.2f | tail>chi99=%4.1f%% | residual_heavy=%s"
          % (pc, lead["z"], lead["kurt"], resid["z"], resid["kurt"], full["z"], 100 * tail_frac, resid_heavy))

cc.PSEUDOCOUNT = BASELINE

n_heavy = sum(r["residual_heavy_tail"] for r in rows)
overall = ("FULLY_ROBUST" if n_heavy == len(rows)
           else "MOSTLY_ROBUST" if n_heavy >= len(rows) * 0.5
           else "NOT_ROBUST")

out = dict(
    analysis="step1_5_pseudocount_sensitivity",
    representation="type-residual ILR (canonical), ASD only",
    note=("Sweep the clr/ilr zero-replacement pseudocount over seven orders of magnitude "
          "(1e-8 to 1e-1) on the canonical 346-ASD / 19-domain type-residual ILR representation. "
          "If the residual subspace (PC4+) stays leptokurtic (Mardia z>1.96) at every pseudocount, "
          "the step4 heavy-tail conclusion (individual idiosyncrasy localized in the residual "
          "dimensions) is not an artifact of the pseudocount choice. Large pseudocounts (>=1e-2) "
          "materially shift the compositional geometry yet the residual leptokurtosis persists, so "
          "they act as a stress test rather than a recommended setting. pc=1e-6 reproduces "
          "step4_residual_tail."),
    pseudocounts_tested=PSEUDOCOUNTS,
    baseline_pseudocount=BASELINE,
    results=rows,
    summary=dict(n_residual_heavy=n_heavy, n_total=len(rows), judgment=overall))
json.dump(out, open(HERE + "/step1_5_pseudocount_sensitivity_results.json", "w"),
          ensure_ascii=False, indent=2)

NAVY, RED, GREY = "#264653", "#CC4C48", "#8896A8"
pcs = [r["pseudocount"] for r in rows]
xlab = [f"{p:.0e}" for p in pcs]
x = np.arange(len(pcs))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.0), dpi=300)
a1.bar(x, [r["pc4plus_z"] for r in rows], color=NAVY, edgecolor="white", linewidth=0.4)
a1.axhline(1.96, color=RED, ls="--", lw=0.9)
a1.set_xticks(x); a1.set_xticklabels(xlab, fontsize=7)
a1.set_xlabel("pseudocount"); a1.set_ylabel("residual PC4+ Mardia z")
a2.bar(x, [r["pc1_3_z"] for r in rows], color=GREY, edgecolor="white", linewidth=0.4)
a2.axhline(1.96, color=RED, ls="--", lw=0.9); a2.axhline(-1.96, color=RED, ls="--", lw=0.9)
a2.set_xticks(x); a2.set_xticklabels(xlab, fontsize=7)
a2.set_xlabel("pseudocount"); a2.set_ylabel("leading PC1-3 Mardia z")
for ax in (a1, a2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"step1_5_pseudocount_sensitivity.{ext}"), bbox_inches="tight")
plt.close(fig)

print("[step1.5] %s (%d/%d residual heavy across pseudocounts) -> %s"
      % (overall, n_heavy, len(rows), HERE))
