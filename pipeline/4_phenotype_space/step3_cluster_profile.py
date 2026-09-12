# -*- coding: utf-8 -*-

import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common_controlled as cc

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


res = {"n_asd": len(P), "n_A": d["n_A"], "n_P": d["n_P"],
       "prototype_top_domains": [(dd, round(float(v), 3)) for dd, v in proto_top],
       "residual_mode_profiles_exploratory": mode_profiles,
       "note": "type-residual; modes exploratory (weak under control)"}
json.dump(res, open(HERE + "/step3_cluster_profile_results.json", "w"), ensure_ascii=False, indent=2)
print("[step3] prototype:", [dd for dd, _ in proto_top])
print("  →", HERE + "/step3_cluster_profile_results.json")
