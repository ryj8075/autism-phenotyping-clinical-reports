#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"

fail=0
run () {
  echo "> $1"
  if python3 "$1"; then echo "  done"; else echo "  FAIL: $1"; fail=1; fi
}

# the confound check runs first; the steps below use its type-residual coordinates
echo "> confound_check/run_all.sh"
if bash confound_check/run_all.sh; then echo "  done"; else echo "  FAIL: confound_check"; fail=1; fi

run step1_effective_dim.py
run step1_5_pseudocount_sensitivity.py
run step2_gmm_vs_heavytail.py
run step3_cluster_profile.py
run step4_residual_tail.py
run step5_mode_deviation.py
run step6_seed_stability.py
run step6_bootstrap_permutation.py
run step7_normative.py
run step8_exemplar.py
run step9_variance.py
run step10_atype_only_mode_stability.py
run step11_ados_variance_share.py

echo "---------------------------------------------------------------"
[ $fail -eq 0 ] && echo "All analyses completed" || echo "Some analyses failed; see FAIL lines above"
