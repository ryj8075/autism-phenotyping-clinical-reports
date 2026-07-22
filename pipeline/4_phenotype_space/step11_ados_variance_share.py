# -*- coding: utf-8 -*-
"""step11 (controlled): phenotype-variance share carried by the ADOS-aligned core domains
(7 DSM-5 ASD-core, group A) vs the domains outside the ADOS/ADI-R construct (B general
child-psychiatric + C format, 12 domains). Current canonical data (346 ASD, 19 domains).

Motivates the 'multidimensional, not reducible to a severity score' point: ADOS/ADI-R totals
summarize the ADOS-construct (A-core) domains, so variance sitting OUTSIDE that construct is
variation those totals cannot represent.

Two variance definitions (they differ, so the manuscript must name which):
  - proportion-space : var of each domain proportion across reports, summed by group / total.
  - CLR-space        : compositional, eigenvalue-weighted squared CLR loadings on the
                       type-residual PCA (the space Figure 4 uses). This is the manuscript number.
"""
import sys, os, json
import numpy as np
from sklearn.decomposition import PCA
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common_controlled as cc

HERE = os.path.dirname(os.path.abspath(__file__))
# group A = 7 ADOS-aligned DSM-5 ASD-core; B = 9 general child-psychiatric; C = 3 format

def share_by_group(per_domain_value, dom):
    g = {"A": 0.0, "B": 0.0, "C": 0.0}
    for i, d in enumerate(dom):
        grp = "A" if i < 7 else ("B" if i < 16 else "C")
        g[grp] += float(per_domain_value[i])
    tot = sum(g.values())
    return {k: round(v / tot, 4) for k, v in g.items()}

def main():
    d = cc.load()
    dom, P, RES = list(d["dom"]), d["P"], d["RES"]

    # proportion-space variance share
    prop = share_by_group(P.var(axis=0, ddof=1), dom)

    # CLR-space variance share on the type-residual PCA (manuscript space)
    p = PCA().fit(RES)
    V = cc.helmert(len(dom))
    load_clr = V @ (p.components_.T * np.sqrt(np.maximum(p.explained_variance_, 0)))  # 19 x 18
    clr_per_domain = np.sum(load_clr ** 2, axis=1)                                    # 19
    clr = share_by_group(clr_per_domain, dom)

    # per-domain CLR-space contribution (% of total), for Figure 4c
    codes = [f"A{i+1}" for i in range(7)] + [f"B{i+1}" for i in range(9)] + \
            [f"C{i+1}" for i in range(len(dom) - 16)]
    tot_clr = float(clr_per_domain.sum())
    per_domain = sorted(
        [dict(domain=dom[i], code=codes[i],
              group=("ADOS-core" if i < 7 else "non-ADOS"),
              clr_variance=float(clr_per_domain[i]),
              contribution_pct=round(100 * clr_per_domain[i] / tot_clr, 2))
         for i in range(len(dom))],
        key=lambda r: -r["contribution_pct"])

    res = {
        "n_asd": int(len(P)), "n_domains": len(dom),
        "group_def": {"A_ADOS_core_7": dom[:7], "B_general_9": dom[7:16], "C_format_3": dom[16:]},
        "proportion_space_share": prop,
        "proportion_space_outside_ADOS": round(prop["B"] + prop["C"], 4),
        "clr_space_share": clr,
        "clr_space_outside_ADOS": round(clr["B"] + clr["C"], 4),
        "per_domain_clr": per_domain,
        "note": ("Manuscript cites the CLR-space (compositional) attribution: ASD-core ~0.43, "
                 "outside-ADOS ~0.57. Proportion-space gives ~0.50/0.50 (different metric). "
                 "per_domain_clr = each domain's share of total CLR variance (Figure 4c input)."),
    }
    out = os.path.join(HERE, "step11_ados_variance_share_results.json")
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"[step11] proportion-space: core {prop['A']:.3f} / outside {res['proportion_space_outside_ADOS']:.3f}")
    print(f"         CLR-space (manuscript): core {clr['A']:.3f} / outside {res['clr_space_outside_ADOS']:.3f}")
    print("  ->", out)

if __name__ == "__main__":
    main()
