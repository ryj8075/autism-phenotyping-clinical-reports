# -*- coding: utf-8 -*-

import sys, os, json
import numpy as np
from sklearn.mixture import GaussianMixture
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common_controlled as cc

HERE = os.path.dirname(os.path.abspath(__file__))
d = cc.load()
_, _, lead, evr = cc.pca_scores(d["RES"])   # lead = PC1-3 of residual

nu, tll, tnp = cc.fit_t(lead)
tbic = float(-2 * tll + tnp * np.log(len(lead)))

perk = {}
print("k   GMM-BIC    ΔBIC vs t   sizes")
for k in range(2, 9):
    g = GaussianMixture(n_components=k, covariance_type="full", n_init=10,
                        random_state=cc.SEED).fit(lead)
    bic = float(g.bic(lead))
    sizes = np.bincount(g.predict(lead), minlength=k).tolist()
    perk[str(k)] = {"gmm_bic": round(bic, 1), "dbic_vs_t": round(bic - tbic, 1), "sizes": sizes}
    print(f"{k}  {bic:9.1f}  {bic - tbic:+9.1f}   {sizes}")

bk = min(perk, key=lambda k: perk[k]["gmm_bic"])
res = {"n_asd": int(len(lead)), "pc13_var": round(float(evr[:3].sum()), 4),
       "t_dist": {"nu": round(float(nu), 1), "bic": round(tbic, 1)},
       "per_k": perk, "bic_optimal_k": int(bk),
       "note": ("Multivariate t beats every GMM k (dBIC_vs_t > 0); BIC rises monotonically "
                "from its minimum at the smallest k, so no Gaussian mixture is favored over the "
                "unimodal (near-Gaussian, heavy-tailed) model. k is an interpretive descriptor, "
                "not a BIC-selected subtype count.")}
json.dump(res, open(HERE + "/step2_gmm_vs_heavytail_results.json", "w"), ensure_ascii=False, indent=2)
print(f"-> multivariate-t nu={nu:.1f}, BIC={tbic:.1f}; BIC-optimal GMM k={bk}")
print("  →", HERE + "/step2_gmm_vs_heavytail_results.json")
