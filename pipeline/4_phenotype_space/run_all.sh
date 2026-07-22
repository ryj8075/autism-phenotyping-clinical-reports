#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"

fail=0
run () {
  echo "> $1"
  if python3 "$1"; then echo "  done"; else echo "  FAIL: $1"; fail=1; fi
}

run step1_effective_dim/step1_effective_dim.py
run step1_5_pseudocount_sensitivity/step1_5_pseudocount_sensitivity.py
run step2_gmm_vs_t/step2_gmm_vs_heavytail.py
run step3_cluster_profile/step3_cluster_profile.py
run step4_residual_tail/step4_residual_tail.py
run step5_mode_deviation/step5_mode_deviation.py
run step5_mode_deviation/step5_mode_deviation_fig.py
run step6_stability/step6_seed_stability.py
run step6_stability/bootstrap_permutation.py
run step7_normative/step7_normative.py
run step8_exemplar/step8_exemplar.py
run step8_exemplar/step8_exemplar_fig.py
run step9_variance/step9_variance.py
run step10_atype_only_mode_stability/atype_only_mode_stability.py

echo "---------------------------------------------------------------"
[ $fail -eq 0 ] && echo "All analyses completed" || echo "Some analyses failed; see FAIL lines above"
