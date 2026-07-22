# -*- coding: utf-8 -*-
"""step9 (controlled): normative model on type-residual ASD-248.
Layer1 = prototypicality (Mahalanobis), Layer2 = per-domain normative deviation z
(vs cohort/prototype, type-controlled CLR space) + MGFS. Writes profiles TSV + JSON."""
import sys, os, json
import numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common_controlled as cc

HERE = os.path.dirname(os.path.abspath(__file__))
d = cc.load()
_, scores, lead, evr = cc.pca_scores(d["RES"])

mu = scores.mean(0); S = np.cov(scores, rowvar=False); Sinv = np.linalg.inv(S)
maha_full = np.sqrt(np.einsum("ij,jk,ik->i", scores - mu, Sinv, scores - mu))
muL = lead.mean(0); SL = np.cov(lead, rowvar=False); SLinv = np.linalg.inv(SL)
maha_lead = np.sqrt(np.einsum("ij,jk,ik->i", lead - muL, SLinv, lead - muL))

axis = np.ones(3) / np.sqrt(3); proj = lead @ axis
mgfs = norm.cdf((proj - proj.mean()) / proj.std(ddof=1)) * 10
CLR_RES = d["CLR_RES"]
zdom = (CLR_RES - CLR_RES.mean(0)) / CLR_RES.std(0, ddof=1)
lab4, post4, _ = cc.gmm_fit(lead, 4)

dom, rid, rtype = d["dom"], d["rid"], d["rtype"]
prof = pd.DataFrame({"report_id": rid, "report_type": rtype, "mgfs": np.round(mgfs, 3),
                     "mahalanobis_full": np.round(maha_full, 3),
                     "mahalanobis_lead3": np.round(maha_lead, 3),
                     "gmm4_mode": lab4, "gmm4_posterior_max": np.round(post4.max(1), 3)})
for j, dd in enumerate(dom):
    prof["z_" + dd] = np.round(zdom[:, j], 3)
prof["max_dev_domain"] = [dom[i] for i in np.argmax(np.abs(zdom), 1)]
prof["max_dev_z"] = np.round(zdom[np.arange(len(zdom)), np.argmax(np.abs(zdom), 1)], 3)
prof.to_csv(HERE + "/controlled_individual_profiles.tsv", sep="\t", index=False)

q90 = float(np.percentile(maha_full, 90))
res = {"n_asd": len(prof), "n_A": d["n_A"], "n_P": d["n_P"],
       "layer1_prototypicality_mahalanobis_full": {
           "mean": round(float(maha_full.mean()), 3), "p90_tail_threshold": round(q90, 3),
           "max": round(float(maha_full.max()), 3), "n_top_decile": int((maha_full >= q90).sum())},
       "mgfs": {"mean": round(float(mgfs.mean()), 3), "sd": round(float(mgfs.std()), 3)},
       "layer2_deviation": "z_<domain> in type-controlled CLR space",
       "profiles_tsv": "controlled_individual_profiles.tsv"}
json.dump(res, open(HERE + "/step7_normative_results.json", "w"), ensure_ascii=False, indent=2)
print("[step7] Mahalanobis mean=%.2f p90=%.2f max=%.2f ; MGFS %.2f±%.2f" %
      (maha_full.mean(), q90, maha_full.max(), mgfs.mean(), mgfs.std()))
print("  →", HERE + "/controlled_individual_profiles.tsv")
