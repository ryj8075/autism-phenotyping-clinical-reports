# -*- coding: utf-8 -*-

import sys, os, json
from itertools import combinations
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common_controlled as cc

HERE = os.path.dirname(os.path.abspath(__file__))
d = cc.load()
RES, dom, V = d["RES"], d["dom"], d["helmert"]   # V = Helmert basis (D x D-1)
rng = np.random.default_rng(cc.SEED)

mu = RES.mean(0)
ci = np.linalg.inv(np.cov(RES, rowvar=False))
diff = RES - mu
mahal = np.sqrt(np.sum(diff @ ci * diff, 1))
thr = np.percentile(mahal, 90)
mask = mahal >= thr

dev = diff[mask]
norms = np.linalg.norm(dev, axis=1, keepdims=True)
norms[norms < 1e-10] = 1e-10
unit = dev / norms
obs = float(np.mean([unit[i] @ unit[j] for i, j in combinations(range(mask.sum()), 2)]))

perm = []
for _ in range(2000):
    idx = rng.choice(len(RES), size=int(mask.sum()), replace=False)
    pdv = diff[idx]
    pn = np.linalg.norm(pdv, axis=1, keepdims=True)
    pn[pn < 1e-10] = 1e-10
    pu = pdv / pn
    perm.append(np.mean([pu[i] @ pu[j] for i, j in combinations(range(len(idx)), 2)]))
perm = np.array(perm)
pval = float((perm >= obs).mean())
null95 = float(np.percentile(perm, 95))
verdict = "DIVERSE_DIRECTION" if obs <= null95 else "SHARED_DIRECTION"

# shared direction mapped ILR -> CLR (domain) space via Helmert basis
mean_dev_clr = V @ dev.mean(0)
order = np.argsort(mean_dev_clr)
top_pos = [(dom[i], round(float(mean_dev_clr[i]), 4)) for i in order[::-1][:5]]
top_neg = [(dom[i], round(float(mean_dev_clr[i]), 4)) for i in order[:5]]

res = {"n_asd": int(len(RES)), "n_outliers": int(mask.sum()),
       "obs_cosine": round(obs, 4), "null95": round(null95, 4), "p_value": pval,
       "verdict": verdict, "shared_dir_pos": top_pos, "shared_dir_neg": top_neg,
       "note": "top-decile Mahalanobis outliers; DIVERSE => idiosyncratic (no single subtype axis)."}
json.dump(res, open(HERE + "/step5_mode_deviation_results.json", "w"), ensure_ascii=False, indent=2)
print("[step5] deviation (top-10%%, n=%d): cosine=%.4f null95=%.4f p=%.4f -> %s" % (
    int(mask.sum()), obs, null95, pval, verdict))
print("  →", HERE + "/step5_mode_deviation_results.json")
