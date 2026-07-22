# -*- coding: utf-8 -*-
"""2_1: sentence-level silver-vs-gold validation (full 489 silver / 26 gold, 19 domains).

For the SAME sentence, compare which domains gold assigned vs which domains the silver
consensus assigned. Silver is zero-shot (the gold label of a sentence was never shown during
labeling); gold is held out for evaluation only. New schema is MULTI-LABEL (up to 3 domains
per sentence), so we build a sentence x domain indicator and report:
  - per-domain Precision/Recall/F1 + support (multilabel; reviewer R4 request) + primary-label confusion
  - per-domain and pooled Spearman (silver confidence vs gold indicator, across sentence x domain cells)
    -> on the SAME footing as the report-level Spearman, so the "report > sentence" claim is comparable.

Report-level references (silver_full_vs_gold_llama_full489_26gold): pooled vector-level Spearman 0.575,
per-domain Spearman medians A=0.70 / B=0.52 / C=0.63, per-report cosine 0.74.
"""
import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get(
    "SILVER_LABELING_DATA_DIR",
    REPO_ROOT / "pipeline" / "2_silver_labeling" / "data",
))
SILVER = Path(os.environ.get(
    "SILVER_LABELS_JSONL",
    DATA / "silver_label" / "silver_labels_26reports.jsonl",
))
GOLD = Path(os.environ.get(
    "GOLD_LABELS_JSONL",
    DATA / "gold_label" / "gold_labels_26reports.jsonl",
))
META = Path(os.environ.get(
    "DOMAIN_META_JSON",
    REPO_ROOT / "pipeline" / "3_multidomain_vectors" / "domain_frequency_vector" /
    "outputs" / "domain_vectors_meta_latest.json",
))
DOMAINS = json.load(open(META))["domain_columns"]
DIDX = {d: i for i, d in enumerate(DOMAINS)}
CORE_A = {"social_emotional_reciprocity", "nonverbal_communication", "relationship_play",
          "stereotyped_behavior", "insistence_on_sameness", "restricted_interests", "sensory_processing"}
FORMAT_C = {"test_scores", "other_general", "recommendations"}
grp = lambda d: "A" if d in CORE_A else ("C" if d in FORMAT_C else "B")

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # index silver by (report_id, sentence_idx): domain -> max consensus confidence
    sil = {}
    for l in open(SILVER):
        if not l.strip():
            continue
        r = json.loads(l)
        conf = {}
        for lab in (r.get("labels") or []):
            d = lab.get("domain_id") if isinstance(lab, dict) else lab
            c = float(lab.get("confidence", 1.0)) if isinstance(lab, dict) else 1.0
            if d in DIDX:
                conf[d] = max(conf.get(d, 0.0), c)
        sil[(r["report_id"], r["sentence_idx"])] = conf

    gold_rows = [json.loads(l) for l in open(GOLD) if l.strip()]
    n_missing = 0
    G, S_bin, S_conf = [], [], []          # sentence x domain
    gold_primary, silver_primary = [], []
    for g in gold_rows:
        key = (g["report_id"], g["sentence_idx"])
        if key not in sil:
            n_missing += 1
            continue
        conf = sil[key]
        gdoms = [d for d in (g.get("labels") or []) if d in DIDX]
        if not gdoms:
            continue
        gv = np.zeros(len(DOMAINS)); sb = np.zeros(len(DOMAINS)); sc = np.zeros(len(DOMAINS))
        for d in gdoms:
            gv[DIDX[d]] = 1
        for d, c in conf.items():
            sb[DIDX[d]] = 1; sc[DIDX[d]] = c
        G.append(gv); S_bin.append(sb); S_conf.append(sc)
        gold_primary.append(gdoms[0])
        silver_primary.append(max(conf, key=conf.get) if conf else "other_general")

    G = np.array(G); S_bin = np.array(S_bin); S_conf = np.array(S_conf)
    n = len(G)

    # per-domain multilabel P/R/F1 (silver presence vs gold presence)
    P, R, F1, sup = precision_recall_fscore_support(G, S_bin, average=None, zero_division=0)
    macro = precision_recall_fscore_support(G, S_bin, average="macro", zero_division=0)[2]
    micro = precision_recall_fscore_support(G, S_bin, average="micro", zero_division=0)[2]
    weighted = precision_recall_fscore_support(G, S_bin, average="weighted", zero_division=0)[2]

    # per-domain Spearman (silver confidence vs gold indicator, across sentences)
    per = []
    for i, d in enumerate(DOMAINS):
        if S_conf[:, i].std() < 1e-12 or G[:, i].std() < 1e-12:
            rho = float("nan")
        else:
            rho = float(spearmanr(S_conf[:, i], G[:, i]).correlation)
        per.append(dict(domain=d, group=grp(d), precision=round(float(P[i]), 3), recall=round(float(R[i]), 3),
                        f1=round(float(F1[i]), 3), support=int(sup[i]), spearman=round(rho, 3)))

    pooled = float(spearmanr(S_conf.flatten(), G.flatten()).correlation)

    def med(field, gg):
        vals = [x[field] for x in per if x["group"] == gg and not (isinstance(x[field], float) and np.isnan(x[field]))]
        return round(float(np.median(vals)), 3) if vals else None
    group_medians = {gg: {"f1": med("f1", gg), "spearman": med("spearman", gg)} for gg in ["A", "B", "C"]}

    cm = confusion_matrix(gold_primary, silver_primary, labels=DOMAINS)

    res = dict(
        timestamp=ts, n_sentences=int(n), n_missing_align=int(n_missing),
        macro_f1=round(float(macro), 3), micro_f1=round(float(micro), 3), weighted_f1=round(float(weighted), 3),
        pooled_spearman_sentence=round(pooled, 3),
        per_domain=per, group_medians=group_medians,
        report_level_reference=dict(pooled_spearman=0.575, per_domain_spearman_median={"A": 0.70, "B": 0.52, "C": 0.63}, per_report_cosine=0.74),
        confusion_primary=dict(labels=DOMAINS, matrix=cm.tolist()),
    )
    out = BASE / "sentence_level_validation_results.json"
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=2)

    print(f"n_sentences={n} (missing align={n_missing}) | macro-F1={macro:.3f} weighted-F1={weighted:.3f}")
    print(f"pooled Spearman  sentence={pooled:.3f}  vs  report={0.575}")
    print(f"per-domain Spearman median  sentence: A={group_medians['A']['spearman']} B={group_medians['B']['spearman']} C={group_medians['C']['spearman']}")
    print(f"                            report:   A=0.70 B=0.52 C=0.63")
    print(f"per-domain F1 median  A={group_medians['A']['f1']} B={group_medians['B']['f1']} C={group_medians['C']['f1']}")
    print("  ->", out)

if __name__ == "__main__":
    main()
