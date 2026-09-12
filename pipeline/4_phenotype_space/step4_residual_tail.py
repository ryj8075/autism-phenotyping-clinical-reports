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
d = cc.load()
_, scores, _, evr = cc.pca_scores(d["RES"])   # scores = all residual PCs
n, ndim = scores.shape

# Mardia heavy-tail decomposition
res = {"n_asd": int(n),
       "pc1_3": cc.mardia_z(scores[:, :3]),
       "pc4plus": cc.mardia_z(scores[:, 3:]),
       "full": cc.mardia_z(d["RES"]),
       "pc13_var": round(float(evr[:3].sum()), 4),
       "pc4plus_var": round(float(evr[3:].sum()), 4)}

# Mahalanobis-to-prototype distribution
mu = scores.mean(0)
ci = np.linalg.inv(np.cov(scores, rowvar=False))
diff = scores - mu
maha = np.sqrt(np.sum(diff @ ci * diff, 1))           # distance to prototype (full residual)
ks = stats.kstest(maha ** 2, "chi2", args=(ndim,))    # d^2 ~ chi2(ndim) under normality
chi99 = np.sqrt(stats.chi2.ppf(0.99, df=ndim))
res["distribution"] = dict(
    ilr_dims=int(ndim),
    mahalanobis=dict(mean=float(maha.mean()), sd=float(maha.std(ddof=1)),
                     median=float(np.median(maha)), max=float(maha.max()),
                     q99=float(np.percentile(maha, 99))),
    ks_d2_vs_chi2=dict(statistic=float(ks.statistic), pvalue=float(ks.pvalue)),
    chi99_ref=float(chi99), tail_frac_beyond_chi99=float(np.mean(maha > chi99)))

res["note"] = ("excess99 = observed 99th-percentile Mahalanobis d^2 / chi-square 99th quantile "
               "(quantile ratio; >1 = tail stretched beyond Gaussian). Positive Mardia z in PC4+ "
               "=> heavy tail localized in residual dims = individual idiosyncrasy. "
               "distribution = Mahalanobis-to-prototype histogram/QQ vs chi(df).")
json.dump(res, open(HERE + "/step4_residual_tail_results.json", "w"), ensure_ascii=False, indent=2)

NAVY, RED, BLUE, GREY = "#264653", "#CC4C48", "#3670B2", "#8896A8"

# histogram + chi(df) reference
fig, ax = plt.subplots(figsize=(3.4, 3.0), dpi=300)
bins = np.linspace(0, max(maha.max(), chi99) * 1.05, 28)
ax.hist(maha, bins=bins, density=True, color=NAVY, alpha=0.80,
        edgecolor="white", linewidth=0.4, label="observed reports")
xx = np.linspace(0, bins[-1], 400)
ax.plot(xx, stats.chi.pdf(xx, df=ndim), color=RED, linewidth=1.6,
        label=f"Gaussian reference\nchi(df={ndim})")
ax.axvline(chi99, color=GREY, linestyle="--", linewidth=0.9)
ax.text(chi99, ax.get_ylim()[1] * 0.45, "  Gaussian 99%", color=GREY,
        fontsize=7, ha="left", va="center")
ax.set_xlabel("Mahalanobis distance to prototype")
ax.set_ylabel("density")
ax.legend(frameon=False, fontsize=7, loc="upper right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"step4_residual_tail_hist.{ext}"), bbox_inches="tight")
plt.close(fig)

# Mahalanobis^2 QQ vs chi^2(df) + KS
d2 = np.sort(maha ** 2)
q = stats.chi2.ppf((np.arange(1, n + 1) - 0.5) / n, df=ndim)
lim = float(max(q.max(), d2.max()))
fig, ax = plt.subplots(figsize=(3.2, 3.0), dpi=300)
ax.plot([0, lim], [0, lim], color=RED, linestyle="--", linewidth=0.9)
ax.scatter(q, d2, s=10, facecolor="white", edgecolor=BLUE, linewidth=0.5, alpha=0.85)
ax.text(lim, lim * 0.02, f"KS vs $\\chi^2$[{ndim}]: p = {ks.pvalue:.1e}",
        fontsize=7, ha="right", va="bottom")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel(f"$\\chi^2_{{{ndim}}}$ quantile")
ax.set_ylabel("observed Mahalanobis$^2$")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"step4_residual_tail_qq.{ext}"), bbox_inches="tight")
plt.close(fig)

print("[step4] tail: PC1-3 z=%.2f(%s) var=%.1f%% | PC4+ z=%.2f(%s) var=%.1f%% | full z=%.2f" % (
    res["pc1_3"]["z"], res["pc1_3"]["kurt"], res["pc13_var"] * 100,
    res["pc4plus"]["z"], res["pc4plus"]["kurt"], res["pc4plus_var"] * 100, res["full"]["z"]))
print("  dist: maha mean=%.2f max=%.2f q99=%.2f | KS(d^2 vs chi2[%d]) p=%.1e | tail>chi99=%.1f%%" % (
    maha.mean(), maha.max(), np.percentile(maha, 99), ndim, ks.pvalue,
    100 * np.mean(maha > chi99)))
print("  →", HERE + "/step4_residual_tail_results.json (+ _hist, _qq figures)")
