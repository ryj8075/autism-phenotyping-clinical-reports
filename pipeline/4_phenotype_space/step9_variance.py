# -*- coding: utf-8 -*-

import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common_controlled as cc

HERE = os.path.dirname(os.path.abspath(__file__))
d = cc.load()
_, _, lead, _ = cc.pca_scores(d["RES"])
RES = d["RES"]
lab4, _, _ = cc.gmm_fit(lead, 4)
lab2, _, _ = cc.gmm_fit(lead, 2)
rng = np.random.default_rng(cc.SEED)

def perm_null(X, labels, n=5000):
    g = X.mean(0); total = ((X - g) ** 2).sum()
    sizes = [np.sum(labels == k) for k in np.unique(labels)]
    idx = np.arange(len(X)); out = []
    for _ in range(n):
        rng.shuffle(idx); lab = np.empty(len(X), int); s = 0
        for ki, sz in enumerate(sizes):
            lab[idx[s:s + sz]] = ki; s += sz
        b = sum(np.sum(lab == ki) * ((X[lab == ki].mean(0) - g) ** 2).sum() for ki in range(len(sizes)))
        out.append(b / total)
    return float(np.mean(out)), float(np.percentile(out, 95))

def decomp(labels, tag):
    _, _, _, e_lead = cc.variance_partition(lead, labels)
    _, _, _, e_full = cc.variance_partition(RES, labels)
    nm, n95 = perm_null(RES, labels)
    return {tag: {"lead3_eta2_between": round(e_lead, 4), "lead3_within": round(1 - e_lead, 4),
                  "full_eta2_between": round(e_full, 4), "full_within": round(1 - e_full, 4),
                  "full_null_eta2_mean": round(nm, 4), "full_null_eta2_p95": round(n95, 4),
                  "mode_sizes": [int(np.sum(labels == k)) for k in np.unique(labels)]}}

res = {"n_asd": len(RES)}
res.update(decomp(lab4, "residual_GMM_k4"))
res.update(decomp(lab2, "residual_GMM_k2_core_vs_periphery"))
# raw reference
_, _, raw_lead, _ = cc.pca_scores(d["ILR"])
lab4_raw, _, _ = cc.gmm_fit(raw_lead, 4)
_, _, _, e_full_raw = cc.variance_partition(d["ILR"], lab4_raw)
res["raw_GMM_k4_full_eta2_between_for_reference"] = round(e_full_raw, 4)
json.dump(res, open(HERE + "/step9_variance_results.json", "w"), ensure_ascii=False, indent=2)
print("[step9] full between: raw k4=%.3f | ctrl k4=%.3f | ctrl 2-way=%.3f | null=%.3f" % (
    e_full_raw, res["residual_GMM_k4"]["full_eta2_between"],
    res["residual_GMM_k2_core_vs_periphery"]["full_eta2_between"],
    res["residual_GMM_k4"]["full_null_eta2_mean"]))
print("  →", HERE + "/step9_variance_results.json")
