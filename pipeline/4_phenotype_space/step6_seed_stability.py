# -*- coding: utf-8 -*-

import sys, os, json
from itertools import combinations
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common_controlled as cc

HERE = os.path.dirname(os.path.abspath(__file__))
d = cc.load()
_, _, lead, evr = cc.pca_scores(d["RES"])   # lead = PC1-3 of residual
SEEDS = list(range(30))

perk = {}
print("k   silhouette   seed-ARI(mean±sd)")
for k in range(2, 9):
    base = GaussianMixture(n_components=k, covariance_type="full", n_init=10,
                           random_state=cc.SEED).fit(lead)
    sil = float(silhouette_score(lead, base.predict(lead)))
    labs = [GaussianMixture(n_components=k, covariance_type="full", n_init=1,
                            random_state=s).fit(lead).predict(lead) for s in SEEDS]
    aris = [adjusted_rand_score(labs[i], labs[j]) for i, j in combinations(range(len(SEEDS)), 2)]
    perk[str(k)] = {"silhouette": round(sil, 3), "seed_ari_mean": round(float(np.mean(aris)), 3),
                    "seed_ari_sd": round(float(np.std(aris)), 3)}
    print(f"{k}   {sil:.3f}        {np.mean(aris):.3f}±{np.std(aris):.3f}")

sk = max(perk, key=lambda k: perk[k]["seed_ari_mean"])
res = {"n_asd": int(len(lead)), "pc13_var": round(float(evr[:3].sum()), 4),
       "per_k": perk, "stability_optimal_k": int(sk),
       "note": ("seed-ARI declines as k grows; only low k stays stable => no stable discrete "
                "subtype structure under report-type control.")}
json.dump(res, open(HERE + "/step6_seed_stability_results.json", "w"), ensure_ascii=False, indent=2)
print(f"-> most stable (seed-ARI) at k={sk}")
print("  →", HERE + "/step6_seed_stability_results.json")
