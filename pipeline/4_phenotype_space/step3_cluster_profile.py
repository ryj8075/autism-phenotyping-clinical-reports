# -*- coding: utf-8 -*-
"""step3 (controlled): prototype domain profile + exploratory residual-mode profiles
+ bootstrap ARI (modes are weak under control). Uses _common_controlled."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common_controlled as cc
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

HERE = os.path.dirname(os.path.abspath(__file__))
d = cc.load()
_, _, lead, _ = cc.pca_scores(d["RES"])
lab4, _, _ = cc.gmm_fit(lead, 4)
P, dom = d["P"], d["dom"]

proto = P.mean(0)
proto_top = sorted(zip(dom, proto), key=lambda x: -x[1])[:6]
mode_profiles = {}
for k in np.unique(lab4):
    mp = P[lab4 == k].mean(0)
    mode_profiles[f"M{k}"] = {"n": int((lab4 == k).sum()),
                              "top_domains": [(dd, round(float(v), 3))
                                              for dd, v in sorted(zip(dom, mp), key=lambda x: -x[1])[:4]]}

# bootstrap ARI: fit PCA+GMM on resample, predict on ORIGINAL, compare to reference labels
rng = np.random.default_rng(cc.SEED)
RES = d["RES"]
aris = []
for _ in range(100):
    bi = rng.integers(0, len(RES), len(RES))
    p = PCA(n_components=RES.shape[1], random_state=cc.SEED).fit(RES[bi])
    g = GaussianMixture(4, covariance_type="full", n_init=3, random_state=cc.SEED).fit(
        p.transform(RES[bi])[:, :3])
    aris.append(adjusted_rand_score(lab4, g.predict(p.transform(RES)[:, :3])))

res = {"n_asd": len(P), "n_A": d["n_A"], "n_P": d["n_P"],
       "prototype_top_domains": [(dd, round(float(v), 3)) for dd, v in proto_top],
       "residual_mode_profiles_exploratory": mode_profiles,
       "bootstrap_ari_mean": round(float(np.mean(aris)), 3),
       "bootstrap_ari_sd": round(float(np.std(aris)), 3),
       "note": "type-residual; modes exploratory (weak under control)"}
json.dump(res, open(HERE + "/step3_cluster_profile_results.json", "w"), ensure_ascii=False, indent=2)
print("[step3] prototype:", [dd for dd, _ in proto_top])
print("[step3] bootstrap ARI %.3f ± %.3f" % (np.mean(aris), np.std(aris)))
print("  →", HERE + "/step3_cluster_profile_results.json")
