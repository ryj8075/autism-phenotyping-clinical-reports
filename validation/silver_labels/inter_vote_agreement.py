#!/usr/bin/env python3
"""T2-(1): 5-vote self-consistency (inter-vote agreement) -- CURRENT canonical data.

Adapted (2026-06-27) from t2_1_inter_vote_agreement.py for the full silver
(silver_labels_26reports.jsonl, 489-cohort schema, 19 domains).

NEW SCHEMA NOTE:
  - Each sentence has raw_responses = 5 votes; each vote now carries a LIST of
    {domain_id, domain_code, confidence, ...} (multi-label), confidence-descending.
  - For per-vote single-category agreement (Fleiss kappa, modal agreement) we take
    each vote's PRIMARY domain = its highest-confidence label (the first one).
    An empty vote (no labels) -> "(none)" abstention category.
  - mean domains per sentence is computed from the aggregated `labels` list.

This stays a SELF-CONSISTENCY metric (same LLM, 5 stochastic runs), not inter-annotator.
"""
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
SILVER = Path(os.environ.get(
    "SILVER_LABELS_JSONL",
    REPO_ROOT / "pipeline" / "2_silver_labeling" / "data" / "silver_label" /
    "silver_labels_26reports.jsonl",
))
META = Path(os.environ.get(
    "DOMAIN_META_JSON",
    REPO_ROOT / "pipeline" / "3_multidomain_vectors" / "3_2_domain_frequency_vector" /
    "outputs" / "domain_vectors_meta_latest.json",
))
DOMAINS = json.load(open(META))["domain_columns"]   # 19 domains
NONE = "(none)"
CATS = DOMAINS + [NONE]                              # 20 categories incl abstention
DIDX = {d: i for i, d in enumerate(CATS)}

def fleiss_kappa(counts):
    """counts: (N, k) integer votes per category per item. Equal n raters per item."""
    N, k = counts.shape
    n = counts.sum(axis=1)
    n_rater = n[0]
    P_i = (np.sum(counts * counts, axis=1) - n_rater) / (n_rater * (n_rater - 1))
    P_bar = P_i.mean()
    p_j = counts.sum(axis=0) / (N * n_rater)
    P_e = np.sum(p_j ** 2)
    return float((P_bar - P_e) / (1 - P_e)) if (1 - P_e) > 0 else float("nan")

def primary_domain(vote):
    """Highest-confidence label of a vote (labels are confidence-descending)."""
    labs = vote.get("labels") or []
    if not labs:
        return NONE
    # robust: pick max confidence (fallback to first)
    best = max(labs, key=lambda x: x.get("confidence", 0) if isinstance(x, dict) else 0)
    dom = best.get("domain_id") if isinstance(best, dict) else best
    return dom if dom in DOMAINS else NONE

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    count_rows = []
    modal_counts = []
    consensus_dom = []
    confidences = []
    n_no_raw = 0
    modal_valid = []
    n_valid_list = []
    none_vote_total = 0
    vote_total = 0
    labels_per_sentence = []

    for line in open(SILVER):
        r = json.loads(line)

        # mean domains per sentence from aggregated labels
        agg = r.get("labels") or []
        labels_per_sentence.append(len(agg))

        rr = r.get("raw_responses")
        if not rr:
            n_no_raw += 1
            continue
        votes = [primary_domain(v) for v in rr]   # 5 primary-domain votes
        vote_total += len(votes)
        none_vote_total += sum(1 for d in votes if d == NONE)
        if len(votes) != 5:
            continue
        vec = np.zeros(len(CATS), dtype=int)
        for v in votes:
            vec[DIDX[v]] += 1
        count_rows.append(vec)
        modal_counts.append(max(Counter(votes).values()))
        valid = [v for v in votes if v != NONE]
        n_valid_list.append(len(valid))
        modal_valid.append(max(Counter(valid).values()) / len(valid) if valid else np.nan)

        # consensus domain = top aggregated label (highest confidence)
        if agg:
            top = max(agg, key=lambda x: x.get("confidence", 0) if isinstance(x, dict) else 0)
            consensus_dom.append(top.get("domain_id") if isinstance(top, dict) else top)
            c = top.get("confidence") if isinstance(top, dict) else None
            if c is not None:
                confidences.append(float(c))
        else:
            consensus_dom.append(None)

    counts = np.array(count_rows)
    modal5 = np.array(modal_counts)
    n = len(modal5)
    kappa = fleiss_kappa(counts)
    dist = Counter(modal5.tolist())
    unanimity = dist.get(5, 0) / n
    majority = (dist.get(4, 0) + dist.get(3, 0)) / n
    split = sum(v for k_, v in dist.items() if k_ < 3) / n
    mean_modal = modal5.mean() / 5
    none_rate = none_vote_total / vote_total
    mv = np.array(modal_valid)
    mv_mean = float(np.nanmean(mv))
    nv = np.array(n_valid_list)
    mean_domains_per_sentence = float(np.mean(labels_per_sentence))

    lines = []

    def p(s=""):
        print(s)
        lines.append(s)

    p(f"=== T2-(1) 5-vote self-consistency (full489 / 19 domains, 26-report silver) ===\n")
    p(f"total sentences: {len(labels_per_sentence)}")
    p(f"sentences (5-vote, none-as-category): {n} (no raw_responses: {n_no_raw})")
    p(f"mean domains per sentence (aggregated labels): {mean_domains_per_sentence:.3f}")
    p(f"  labels-per-sentence dist: {dict(sorted(Counter(labels_per_sentence).items()))}")
    p(f"vote-level abstention(None) rate: {none_rate:.3f} ({none_vote_total}/{vote_total})")
    p(f"valid-votes-per-sentence: all5={int((nv == 5).sum())} "
      f"dist={dict(sorted(Counter(nv.tolist()).items(), reverse=True))}")
    p(f"Fleiss' kappa (5 rater x {len(CATS)} cat incl none): {kappa:.3f}")
    p(f"mean modal agreement (incl none): {mean_modal:.3f} ({modal5.mean():.2f}/5)")
    p(f"mean modal agreement (among valid votes only): {mv_mean:.3f}")
    p(f"unanimity(5/5): {unanimity:.3f}  majority(3-4): {majority:.3f}  split(<3): {split:.3f}")
    p(f"modal count dist (out of 5, incl none): {dict(sorted(dist.items(), reverse=True))}")
    if confidences:
        cf = np.array(confidences)
        p(f"consensus (top-label) confidence: mean={cf.mean():.3f} median={np.median(cf):.3f} "
          f"min={cf.min():.3f} max={cf.max():.3f}")

    cons = np.array([c if c is not None else NONE for c in consensus_dom])
    per_dom = {}
    p(f"\n{'domain':<28}{'n':>6}{'mean_modal':>12}{'unanimity':>11}")
    for d in DOMAINS:
        m = cons == d
        if m.sum() == 0:
            per_dom[d] = dict(n=0, mean_modal=None, unanimity=None)
            continue
        mm = modal5[m]
        per_dom[d] = dict(n=int(m.sum()), mean_modal=float(mm.mean() / 5),
                          unanimity=float((mm == 5).mean()))
        p(f"{d:<28}{int(m.sum()):>6}{mm.mean() / 5:>12.3f}{(mm == 5).mean():>11.3f}")

    out = dict(
        timestamp=ts, schema="full489_26gold", n_total_sentences=len(labels_per_sentence),
        n_sentences_5vote=int(n), n_no_raw=int(n_no_raw),
        mean_domains_per_sentence=mean_domains_per_sentence,
        labels_per_sentence_dist={int(k): int(v) for k, v in Counter(labels_per_sentence).items()},
        vote_abstention_none_rate=float(none_rate),
        fleiss_kappa_incl_none=kappa, n_categories=len(CATS),
        mean_modal_agreement_incl_none=float(mean_modal),
        mean_modal_agreement_valid_only=mv_mean,
        unanimity_rate=float(unanimity), majority_rate=float(majority), split_rate=float(split),
        valid_votes_per_sentence={int(k): int(v) for k, v in Counter(nv.tolist()).items()},
        modal_count_distribution={int(k): int(v) for k, v in dist.items()},
        confidence=(dict(mean=float(np.mean(confidences)), median=float(np.median(confidences)),
                         min=float(np.min(confidences)), max=float(np.max(confidences)))
                    if confidences else None),
        per_domain=per_dom,
        note=("Per-vote primary domain = highest-confidence label of the 5-vote multi-label "
              "responses; empty vote -> (none). Same-LLM 5-run self-consistency, not "
              "inter-annotator. mean_domains_per_sentence from aggregated labels."))
    out_json = BASE / f"t2_1_inter_vote_agreement_full489_26gold_{ts}.json"
    json.dump(out, open(out_json, "w"), indent=2, ensure_ascii=False)
    (BASE / "t2_1_inter_vote_agreement_full489_26gold.txt").write_text("\n".join(lines) + "\n")
    p(f"\n[saved] {out_json}")
    p(f"[saved] {BASE / 't2_1_inter_vote_agreement_full489_26gold.txt'}")

if __name__ == "__main__":
    main()
