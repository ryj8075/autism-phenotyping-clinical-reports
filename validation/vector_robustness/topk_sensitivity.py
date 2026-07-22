from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

_THIS_DIR = Path(__file__).resolve().parent
_COMMON_DIR = _THIS_DIR.parent / "_common"
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))

_REPO_ROOT = _THIS_DIR.parents[1]
_CC_DIR = str(Path(os.environ.get(
    "PHENOTYPE_SPACE_DIR",
    _REPO_ROOT / "pipeline" / "4_phenotype_space",
)))
if _CC_DIR not in sys.path:
    sys.path.insert(0, _CC_DIR)
import _common_controlled as cc  # noqa: E402

from vector_builder import (  # noqa: E402
    build_proportion_matrix,
    compute_valid_sentence_mask,
    extract_top_k_indices,
    load_active_domains,
    load_silver_labels,
)
from idiosyncrasy_checks import run_all_checks  # noqa: E402

N_SEED_CROSS = 30

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_tokenized_dir(tokenized_dir: Path):

    mask = torch.load(tokenized_dir / "attention_mask_tensor",
                      map_location="cpu", weights_only=False)
    rids = torch.load(tokenized_dir / "report_id_array",
                      map_location="cpu", weights_only=False)
    lab = torch.load(tokenized_dir / "label_tensor",
                     map_location="cpu", weights_only=False)
    if hasattr(mask, "numpy"):
        mask = mask.numpy()
    else:
        mask = np.asarray(mask)
    lab = lab.numpy() if hasattr(lab, "numpy") else np.asarray(lab)
    return mask, np.asarray(rids), lab

def report_ptype(report_id: str) -> str:

    r = str(report_id)
    return "A" if "-A" in r else ("P" if "-P" in r else "?")

def _gmm(k: int, seed: int, n_init: int) -> GaussianMixture:
    return GaussianMixture(n_components=k, covariance_type="full", n_init=n_init,
                           max_iter=500, random_state=seed)

def heavytail_ablation_convention(RES: np.ndarray) -> Dict:

    _, scores, _lead, evr = cc.pca_scores(RES)
    full = cc.mardia_z(RES)
    lead_m = cc.mardia_z(scores[:, :3])
    resid_m = cc.mardia_z(scores[:, 3:])
    pattern = bool(lead_m["z"] < 0 and resid_m["z"] > 0)
    return dict(
        n_ilr_dim=int(RES.shape[1]),
        full_beta2=round(full["beta2"], 1), full_expected=full["expected"],
        full_z=round(full["z"], 1), full_kurt=full["kurt"],
        full_excess99=round(full["excess99"], 3),
        lead3_z=round(lead_m["z"], 2), lead3_kurt=lead_m["kurt"],
        lead3_var_frac=round(float(evr[:3].sum()), 3),
        resid_z=round(resid_m["z"], 2), resid_kurt=resid_m["kurt"],
        resid_var_frac=round(float(evr[3:].sum()), 3),
        pattern_preserved=pattern,
    )

def cross_seed_ari(lead: np.ndarray, k: int, n_seed: int = N_SEED_CROSS) -> float:

    labs = [GaussianMixture(k, covariance_type="full", n_init=1, random_state=s,
                            max_iter=500).fit(lead).predict(lead) for s in range(n_seed)]
    a = [adjusted_rand_score(labs[i], labs[j])
         for i in range(n_seed) for j in range(i + 1, n_seed)]
    return float(np.mean(a))

def cluster_stability_ari(lead: np.ndarray, k: int, seed: int = 42,
                          n_seeds: int = 30, n_boot: int = 1000) -> Dict:

    n = lead.shape[0]
    labs = [_gmm(k, s, 1).fit(lead).predict(lead) for s in range(n_seeds)]
    seed_aris = [adjusted_rand_score(labs[i], labs[j])
                 for i, j in combinations(range(n_seeds), 2)]

    rng = np.random.default_rng(seed)
    ref = _gmm(k, seed, 10).fit(lead).predict(lead)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = adjusted_rand_score(ref, _gmm(k, b, 10).fit(lead[idx]).predict(lead))
    return {
        "k": int(k),
        "seed_ari_mean": float(np.mean(seed_aris)),
        "seed_ari_sd": float(np.std(seed_aris)),
        "bootstrap_ari_mean": float(boot.mean()),
        "bootstrap_ari_ci95_lo": float(np.percentile(boot, 2.5)),
        "bootstrap_ari_ci95_hi": float(np.percentile(boot, 97.5)),
    }

def strip_arrays(check_results: Dict) -> Dict:
    return {k: v for k, v in check_results.items() if k != "_arrays"}

def summarize_k(k: int, check: Dict, stats: Dict, ht: Dict, ari2: float,
                ari4: float) -> Dict:

    eig = check["B_eigenvalue"]
    gmm = check["C_gmm_vs_t"]
    mgfs = check["E_mgfs"]
    out = {
        "K": int(k),
        "match_rate": float(stats.get("match_rate", np.nan)),
        "zero_rows": int(stats.get("zero_rows", 0)),

        "n_ilr_dim": ht["n_ilr_dim"],
        "full_beta2": ht["full_beta2"],
        "full_expected": ht["full_expected"],
        "full_z": ht["full_z"],
        "full_kurt": ht["full_kurt"],
        "full_excess99": ht["full_excess99"],
        "lead3_z": ht["lead3_z"],
        "lead3_kurt": ht["lead3_kurt"],
        "lead3_var_frac": ht["lead3_var_frac"],
        "resid_z": ht["resid_z"],
        "resid_kurt": ht["resid_kurt"],
        "resid_var_frac": ht["resid_var_frac"],
        "pattern_preserved": ht["pattern_preserved"],

        "cross_seed_ari_k2": round(ari2, 3),
        "cross_seed_ari_k4": round(ari4, 3),

        "kaiser_pcs": int(eig["n_kaiser"]),
        "effective_dim": float(eig["effective_dim"]),
        "pc1_variance": float(eig["pc1_variance"]),
        "gmm_vs_t_winner": gmm["winner"],
        "best_gmm_k": int(gmm["best_gmm_k"]),
        "t_nu": float(gmm["t_nu"]),
        "mgfs_mean": float(mgfs["mgfs_mean"]),
        "mgfs_std": float(mgfs["mgfs_std"]),
    }
    return out

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Track 1: Top-K sensitivity (ablation-convention heavy tail)")
    parser.add_argument("--attention_matrix", type=str, required=True)
    parser.add_argument("--tokenized_dir", type=str, required=True)
    parser.add_argument("--silver_labels", type=str, required=True)
    parser.add_argument("--domains_yaml", type=str, required=True)
    parser.add_argument("--k_values", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--attention_method", type=str, default="column_sum",
                        choices=["column_sum", "row_sum"])
    parser.add_argument("--pilot_mode", action="store_true")
    parser.add_argument("--asd_only", action="store_true", default=True)
    parser.add_argument("--all_labels", dest="asd_only", action="store_false")
    parser.add_argument("--residualize_ptype", action="store_true", default=True)
    parser.add_argument("--no_residualize", dest="residualize_ptype", action="store_false")
    parser.add_argument("--n_perm", type=int, default=1000)
    parser.add_argument("--no_ari", action="store_true",
                        help="Skip bootstrap ARI for the best_gmm_k split. Cross-seed ARI for k=2/k=4 is still computed.")
    parser.add_argument("--ari_n_boot", type=int, default=1000)
    parser.add_argument("--verify_canonical", action="store_true", default=True,
                        help="Compare K=10 residualized scores against canonical cc.load() outputs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=str(_THIS_DIR / "results"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[1] Loading attention matrix: {args.attention_matrix}")
    attn = np.load(args.attention_matrix)
    print(f"      shape: {attn.shape}")

    print(f"[1] Loading tokenized dir: {args.tokenized_dir}")
    mask, rids, labels = load_tokenized_dir(Path(args.tokenized_dir))
    print(f"      mask shape: {mask.shape}, n_reports: {len(rids)}")

    ptype = np.array([report_ptype(r) for r in rids])
    asd_mask = (np.asarray(labels) == 1)
    print(f"      ASD(label==1): {int(asd_mask.sum())}, non-ASD: {int((~asd_mask).sum())}"
          f" | report type A/P: {int((ptype=='A').sum())}/{int((ptype=='P').sum())}")
    print(f"      asd_only={args.asd_only}, residualize_ptype={args.residualize_ptype}")

    if attn.shape[0] != len(rids):
        raise ValueError(
            f"attention N={attn.shape[0]} != report_id_array length {len(rids)}")

    valid_mask = compute_valid_sentence_mask(mask)

    print(f"[1] Loading silver labels: {args.silver_labels}")
    silver = load_silver_labels(args.silver_labels)
    print(f"      {len(silver)} labeled sentences")

    print(f"[1] Loading domains: {args.domains_yaml}")
    active_domains = load_active_domains(args.domains_yaml, pilot_mode=args.pilot_mode)
    print(f"      {len(active_domains)} active domains")

    canon = None
    if args.verify_canonical:
        try:
            canon = cc.load()
            print(f"[1] canonical loaded: n_asd={len(canon['P'])}, "
                  f"n_dom={len(canon['dom'])}")
        except Exception as e:  # pragma: no cover
            print(f"[1] canonical load failed ({e}); verification skipped")
            canon = None

    all_results: Dict[str, Dict] = {}
    summaries: List[Dict] = []
    verification: Dict = {}

    for k in args.k_values:
        print(f"\n[1] === K = {k} ===")
        top_idx = extract_top_k_indices(attn, top_k=k, method=args.attention_method,
                                        valid_mask=valid_mask)
        X_prop, stats = build_proportion_matrix(top_idx, rids, silver, active_domains)
        print(f"      match_rate: {stats['match_rate']:.3f}, zero_rows: {stats['zero_rows']}")

        if args.asd_only:
            X_used = X_prop[asd_mask]
            ptype_used = ptype[asd_mask]
        else:
            X_used = X_prop
            ptype_used = ptype
        stats["n_used"] = int(X_used.shape[0])
        stats["asd_only"] = bool(args.asd_only)
        stats["residualize_ptype"] = bool(args.residualize_ptype)
        resid_arg = ptype_used if args.residualize_ptype else None

        design = np.column_stack([np.ones(len(X_used)),
                                  (ptype_used == "P").astype(float)])
        RES = cc.residualize(cc.ilr(X_used), design) if args.residualize_ptype \
            else cc.ilr(X_used)

        checks = run_all_checks(X_used, n_perm_direction=args.n_perm, seed=args.seed,
                                residualize_ptype=resid_arg)

        max_abs_diff = float(np.abs(checks["_arrays"]["X_ilr"] - RES).max())
        stats["res_vs_cc_max_abs_diff"] = max_abs_diff

        ht = heavytail_ablation_convention(RES)
        print(f"      full z={ht['full_z']} ({ht['full_kurt']}) | "
              f"lead3 z={ht['lead3_z']} ({ht['lead3_kurt']}) | "
              f"resid z={ht['resid_z']} ({ht['resid_kurt']}) | "
              f"pattern={ht['pattern_preserved']}")

        _, _scores, lead_cc, _evr = cc.pca_scores(RES)
        ari2 = cross_seed_ari(lead_cc, 2)
        ari4 = cross_seed_ari(lead_cc, 4)
        print(f"      cross-seed ARI: k2={ari2:.3f}, k4={ari4:.3f}")

        stab = None
        if not args.no_ari:
            best_k = int(checks["C_gmm_vs_t"]["best_gmm_k"])
            lead = checks["_arrays"]["pca_scores"][:, :3]
            stab = cluster_stability_ari(lead, best_k, seed=args.seed,
                                         n_boot=args.ari_n_boot)
            checks["C_stability_bestk"] = stab

        checks["A_heavytail_subspaces"] = ht
        checks["C_cross_seed_ari"] = {"k2": ari2, "k4": ari4,
                                      "n_seeds": N_SEED_CROSS}

        all_results[f"K={k}"] = {
            "build_stats": stats,
            "checks": strip_arrays(checks),
        }
        summary = summarize_k(k, checks, stats, ht, ari2, ari4)
        if stab is not None:
            summary["bestk_seed_ari"] = round(stab["seed_ari_mean"], 3)
            summary["bestk_bootstrap_ari"] = round(stab["bootstrap_ari_mean"], 3)
            summary["bestk_bootstrap_ari_ci95"] = [round(stab["bootstrap_ari_ci95_lo"], 3),
                                                   round(stab["bootstrap_ari_ci95_hi"], 3)]
        summaries.append(summary)

        print(f"      Kaiser PCs={summary['kaiser_pcs']}, "
              f"eff_dim={summary['effective_dim']:.2f}, "
              f"PC1={summary['pc1_variance']:.3f}")
        print(f"      GMM vs t: {summary['gmm_vs_t_winner']} "
              f"(best_gmm_k={summary['best_gmm_k']}, nu={summary['t_nu']:.1f})")

        if canon is not None and k == 10:
            Pc = canon["P"]
            same_shape = (Pc.shape == X_used.shape)
            v = {"canonical_shape": list(Pc.shape), "topk_shape": list(X_used.shape),
                 "same_shape": bool(same_shape)}
            if same_shape:

                order = {str(r): i for i, r in enumerate(canon["rid"])}
                rid_used = [str(r) for r in np.asarray(rids)[asd_mask]] \
                    if args.asd_only else [str(r) for r in rids]
                if all(r in order for r in rid_used):
                    perm = [order[r] for r in rid_used]
                    v["prop_max_abs_diff"] = float(np.abs(Pc[perm] - X_used).max())
                    RES_c = canon["RES"][perm]
                    v["res_max_abs_diff"] = float(np.abs(RES_c - RES).max())
                    v["canonical_heavytail"] = heavytail_ablation_convention(canon["RES"])
                else:
                    v["id_match"] = False
            verification["K=10_vs_canonical"] = v
            print(f"      [verify] K=10 vs canonical: {v}")

    out_path = output_dir / f"topk_sensitivity_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": {
            "timestamp": ts,
            "script": "topk_sensitivity.py",
            "convention": "ablation (step2_heavytail_sensitivity + drop_domains_robustness)",
            "k_values": list(args.k_values),
            "attention_matrix": str(args.attention_matrix),
            "silver_labels": str(args.silver_labels),
            "domains_yaml": str(args.domains_yaml),
            "n_reports": int(len(rids)),
            "asd_only": bool(args.asd_only),
            "n_used": int(asd_mask.sum()) if args.asd_only else int(len(rids)),
            "residualize_ptype": bool(args.residualize_ptype),
            "n_seeds_cross": N_SEED_CROSS,
            "seed": int(args.seed),
        }, "verification": verification, "per_k": all_results}, f,
            ensure_ascii=False, indent=2,
            default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))

    summary_path = output_dir / f"topk_summary_{ts}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"seed": int(args.seed),
                   "n_asd": int(asd_mask.sum()) if args.asd_only else int(len(rids)),
                   "n_seeds_cross": N_SEED_CROSS,
                   "verification": verification,
                   "summaries": summaries}, f, ensure_ascii=False, indent=2)

    print("\n[1] ===== Aggregate decision =====")
    print(f"  pattern_preserved for all K: {all(s['pattern_preserved'] for s in summaries)}")
    print(f"  resid_z by K: {[(s['K'], s['resid_z']) for s in summaries]}")
    print(f"  lead3_z by K: {[(s['K'], s['lead3_z']) for s in summaries]}")
    print(f"  eff_dim by K: {[(s['K'], round(s['effective_dim'], 2)) for s in summaries]}")
    print(f"  ARI@k2 by K:  {[(s['K'], s['cross_seed_ari_k2']) for s in summaries]}")
    print("\nSaved outputs:")
    print(f"  full:    {out_path}")
    print(f"  summary: {summary_path}")

if __name__ == "__main__":
    main()
