# -*- coding: utf-8 -*-

import sys, json
from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))          # 2_2 root for _sensitivity_common
import _sensitivity_common as sc
cc = sc.cc
N_SEED = 30

def eff_dim(evr):
    p = evr / evr.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))

def cross_seed_ari(lead, k):
    labs = [GaussianMixture(k, covariance_type="full", n_init=1, random_state=s,
                            max_iter=500).fit(lead).predict(lead) for s in range(N_SEED)]
    a = [adjusted_rand_score(labs[i], labs[j]) for i in range(N_SEED) for j in range(i + 1, N_SEED)]
    return float(np.mean(a))

def headline(RES, ilr_dim):
    _, _, lead, evr = cc.pca_scores(RES)
    nu, tll, tnp = cc.fit_t(lead)
    tbic = float(-2 * tll + tnp * np.log(len(lead)))
    gbic = {k: float(GaussianMixture(k, covariance_type="full", n_init=10, random_state=cc.SEED,
                                     max_iter=500).fit(lead).bic(lead)) for k in range(2, 9)}
    bk = min(gbic, key=gbic.get)
    return dict(eff_dim=round(eff_dim(evr), 2), eff_dim_frac=round(eff_dim(evr) / ilr_dim, 3),
                pc1_evr=round(float(evr[0]), 3), pc13_evr=round(float(evr[:3].sum()), 3),
                t_bic=round(tbic, 1), best_gmm_k=bk, best_gmm_bic=round(gbic[bk], 1),
                dbic_t_minus_gmm=round(gbic[bk] - tbic, 1), t_wins=bool(gbic[bk] > tbic),
                cross_seed_ari_k2=round(cross_seed_ari(lead, 2), 3),
                cross_seed_ari_k4=round(cross_seed_ari(lead, 4), 3))

def main():
    d = cc.load()
    variants, dom = sc.build_variants(d)
    names = ["baseline_full19", "downweight_sentence_f1", "downweight_report_rho"]
    res = {"seed": cc.SEED, "n_asd": int(len(d["P"])), "n_seeds_cross": N_SEED, "variants": {}}
    for name in names:
        v = variants[name]
        r = headline(v["RES"], v["ilr_dim"])
        r["kind"] = v["kind"]
        r["n_domains"] = v["n_domains"]
        if v["kind"] == "downweight":
            r["weights"] = v["weights"]
        res["variants"][name] = r
        print(f"[{name}] kind={v['kind']} eff_dim={r['eff_dim']} (frac {r['eff_dim_frac']}) "
              f"PC1={r['pc1_evr']} t_wins={r['t_wins']} (dBIC {r['dbic_t_minus_gmm']}, bestk={r['best_gmm_k']}) "
              f"ARI k2={r['cross_seed_ari_k2']} k4={r['cross_seed_ari_k4']}")
    out = BASE / "downweight_robustness_results.json"
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=2)
    print("  ->", out)

if __name__ == "__main__":
    main()
