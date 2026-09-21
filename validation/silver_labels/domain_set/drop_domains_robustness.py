# -*- coding: utf-8 -*-

import sys, os, json
from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

REPO_ROOT = Path(__file__).resolve().parents[3]
CC = str(Path(os.environ.get(
    "PHENOTYPE_SPACE_DIR",
    REPO_ROOT / "pipeline" / "4_phenotype_space",
)))
sys.path.insert(0, CC)
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import _common_controlled as cc
import _sensitivity_common as sc

SEED = cc.SEED
N_SEED = 30

def eff_dim(evr):
    p = evr / evr.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))

def cross_seed_ari(lead, k):
    labs = [GaussianMixture(k, covariance_type="full", n_init=1, random_state=s,
                            max_iter=500).fit(lead).predict(lead) for s in range(N_SEED)]
    a = [adjusted_rand_score(labs[i], labs[j]) for i in range(N_SEED) for j in range(i + 1, N_SEED)]
    return float(np.mean(a))

def analyze(P, dom, design, keep):
    idx = [i for i, d in enumerate(dom) if d in keep]
    Psub = P[:, idx]                      # clr() is scale-invariant and adds a pseudocount, so
    RES = cc.residualize(cc.ilr(Psub), design)   # unnormalized rows (incl. all-dropped -> all-zero) are fine
    _, _, lead, evr = cc.pca_scores(RES)
    # GMM vs t on leading PC1-3
    nu, tll, tnp = cc.fit_t(lead)
    tbic = float(-2 * tll + tnp * np.log(len(lead)))
    gbic = {k: float(GaussianMixture(k, covariance_type="full", n_init=10, random_state=SEED,
                                     max_iter=500).fit(lead).bic(lead)) for k in range(2, 9)}
    bk = min(gbic, key=gbic.get)
    return dict(n_domains=len(idx), ilr_dim=len(idx) - 1,
                eff_dim=round(eff_dim(evr), 2), eff_dim_frac=round(eff_dim(evr) / (len(idx) - 1), 3),
                pc1_evr=round(float(evr[0]), 3), pc13_evr=round(float(evr[:3].sum()), 3),
                t_bic=round(tbic, 1), best_gmm_k=bk, best_gmm_bic=round(gbic[bk], 1),
                dbic_t_minus_gmm=round(gbic[bk] - tbic, 1), t_wins=bool(gbic[bk] > tbic),
                cross_seed_ari_k2=round(cross_seed_ari(lead, 2), 3),
                cross_seed_ari_k4=round(cross_seed_ari(lead, 4), 3),
                bestk_stability=sc.bestk_stability(lead, bk))

def main():
    d = cc.load()
    P, dom, design = d["P"], list(d["dom"]), np.column_stack(
        [np.ones(len(d["P"])), (d["rtype"] == "P").astype(float)])
    f1, rho = sc.fidelity()

    drops = {
        "baseline_full19": set(),
        "drop_sentence_f1_lt_0.3": {dm for dm, v in f1.items() if v < 0.3},
        "drop_report_rho_lt_0.3": {dm for dm, v in rho.items() if v < 0.3},
        "drop_report_rho_lt_0": {dm for dm, v in rho.items() if v < 0},
    }
    res = {"seed": SEED, "n_asd": int(len(P)), "n_seeds_cross": N_SEED, "variants": {}}
    for name, drop in drops.items():
        keep = [dm for dm in dom if dm not in drop]
        r = analyze(P, dom, design, set(keep))
        r["dropped"] = sorted(drop)
        res["variants"][name] = r
        print(f"[{name}] kept={r['n_domains']}  eff_dim={r['eff_dim']} (frac {r['eff_dim_frac']})  "
              f"PC1={r['pc1_evr']}  t_wins={r['t_wins']} (dBIC {r['dbic_t_minus_gmm']}, bestk={r['best_gmm_k']})  "
              f"ARI k2={r['cross_seed_ari_k2']} k4={r['cross_seed_ari_k4']}")
        if drop:
            print(f"        dropped({len(drop)}): {sorted(drop)}")
    out = BASE / "drop_domains_robustness_results.json"
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=2)
    print("  ->", out)

if __name__ == "__main__":
    main()
