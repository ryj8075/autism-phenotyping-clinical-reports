# -*- coding: utf-8 -*-
"""step10 (controlled): exemplar candidates per residual mode, from the controlled
step9 profiles. (For the journal Fig 6e a mode-free exemplar is chosen separately;
this keeps the per-mode candidate listing for reference.)"""
import os, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STEP9_TSV = os.path.join(os.path.dirname(HERE), "step7_normative",
                         "controlled_individual_profiles.tsv")

prof = pd.read_csv(STEP9_TSV, sep="\t")
zcols = [c for c in prof.columns if c.startswith("z_")]
dom = [c[2:] for c in zcols]
lab = prof["gmm4_mode"].values
maha = prof["mahalanobis_full"].values
post = prof["gmm4_posterior_max"].values
rid = prof["report_id"].values
zdom = prof[zcols].values

exemplars = {}
for k in np.unique(lab):
    mz = zdom[lab == k].mean(0)
    sig = int(np.argmax(np.abs(mz)))
    m = lab == k
    cand = pd.DataFrame({"report_id": rid[m], "posterior": post[m],
                         "mahalanobis": maha[m], "within_z_signal": zdom[m, sig]})
    cand = cand[cand.posterior > 0.85]
    cand = cand.reindex(cand["within_z_signal"].abs().sort_values(ascending=False).index).head(5)
    exemplars[f"M{k}"] = {"signal_domain": dom[sig],
                          "top": [{"report_id": r.report_id, "posterior": round(r.posterior, 3),
                                   "mahalanobis": round(r.mahalanobis, 3),
                                   "within_z": round(r.within_z_signal, 3)} for r in cand.itertuples()]}

json.dump({"exemplars_per_residual_mode": exemplars,
           "note": "modes exploratory; journal Fig 6e uses a mode-free exemplar "
                   "(top-decile Mahalanobis x reliable ASD-core peak)"},
          open(HERE + "/step8_exemplar_results.json", "w"), ensure_ascii=False, indent=2)
print("[step10] exemplar candidates per residual mode:")
for k, v in exemplars.items():
    t = v["top"][0] if v["top"] else None
    print(f"  {k} signal={v['signal_domain']}: " +
          (f"{t['report_id']} post={t['posterior']} maha={t['mahalanobis']} z={t['within_z']}"
           if t else "none>0.85"))
print("  →", HERE + "/step8_exemplar_results.json")
