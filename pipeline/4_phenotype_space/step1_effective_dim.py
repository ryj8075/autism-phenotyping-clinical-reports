# -*- coding: utf-8 -*-

import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common_controlled as cc

HERE = os.path.dirname(os.path.abspath(__file__))
d = cc.load()
pca, _, _, evr = cc.pca_scores(d["RES"])
eig = pca.explained_variance_

p = eig / eig.sum()                                            # normalized eigenvalues
shannon_eff_dim = float(np.exp(-np.sum(p * np.log(p))))        # exp(Shannon entropy) = perplexity (q=1)
participation_ratio = float((eig.sum() ** 2) / np.sum(eig ** 2))  # inverse Simpson (q=2)
kaiser = int(np.sum(eig >= eig.mean()))
top3_var = float(evr[:3].sum())

def shannon_effdim(M):
    """exp(Shannon entropy of normalized PCA eigenvalues) for representation M."""
    e = cc.pca_scores(M)[0].explained_variance_
    q = e / e.sum()
    return float(np.exp(-np.sum(q * np.log(q))))

rtype = d["rtype"]
eff_raw = shannon_effdim(d["ILR"])                       # raw ILR, no type control
eff_atype = shannon_effdim(d["ILR"][rtype == "A"])       # autism-diagnostic reports only

res = {"n_asd": int(len(d["RES"])), "n_A": d["n_A"], "n_P": d["n_P"],
       "n_ilr_dims": int(len(eig)),
       "shannon_entropy_eff_dim": round(shannon_eff_dim, 2),
       "participation_ratio_eff_dim": round(participation_ratio, 2),
       "kaiser_guttman_dims": kaiser,
       "top3_pc_var": round(top3_var, 4),
       "shannon_eff_dim_robustness": {
           "type_residual": round(shannon_eff_dim, 2),         # primary
           "raw": round(eff_raw, 2),
           "Atype_only": round(eff_atype, 2),
           "n_Atype": int((rtype == "A").sum())},
       "note": ("type-residual ASD; high effective dimensionality => representation spread "
                "across many dimensions (idiosyncrasy), not a low-rank subtype structure. "
                "shannon_entropy_eff_dim (exp of Shannon entropy of normalized eigenvalues) is "
                "the manuscript/Figure-4 metric; participation_ratio (inverse Simpson) weights "
                "dominant components more and is always <= the entropy one. "
                "shannon_eff_dim_robustness reports the same metric raw / type-residual / "
                "A-type-only.")}
json.dump(res, open(HERE + "/step1_effective_dim_results.json", "w"), ensure_ascii=False, indent=2)
print("[step1] eff-dim: Shannon=%.2f | participation ratio=%.2f / %d dims ; Kaiser=%d ; top3 PC var=%.1f%%"
      % (shannon_eff_dim, participation_ratio, len(eig), kaiser, top3_var * 100))
print("[step1] robustness: residual=%.2f | raw=%.2f | A-type-only=%.2f (n_A=%d)"
      % (shannon_eff_dim, eff_raw, eff_atype, int((rtype == "A").sum())))
print("  →", HERE + "/step1_effective_dim_results.json")
