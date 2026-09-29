#!/usr/bin/env bash
# Actual full training: identical complete V5 recipe and validation stopping rule.
# Run 282 and 283 first. Resume with this same command; raw batches and epoch
# checkpoints validate their identities before reuse. Keep this launcher stable
# while running because bash reads its source incrementally.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export NUMBA_NUM_THREADS=1
export PYTHONHASHSEED=0
PY=${MAGICC_R7_PYTHON:-python}
R7_OUT=results/revision/holdout_resubmission7
mkdir -p "$R7_OUT/logs"
printf '%s\n' "$$" > "$R7_OUT/pipeline.pid"
"$PY" scripts/290_audit_resubmission7_holdout.py --preflight
run_arm() {
    local variant=$1
    "$PY" scripts/284_build_resubmission7_holdout_data.py --variant "$variant" --workers 18
    "$PY" scripts/285_train_resubmission7_holdout.py --variant "$variant" --threads 12
}
run_arm holdout > "$R7_OUT/logs/holdout_full_pipeline.log" 2>&1 &
HO_PID=$!
run_arm matched_full > "$R7_OUT/logs/matched_full_pipeline.log" 2>&1 &
MF_PID=$!
trap 'kill "$HO_PID" "$MF_PID" 2>/dev/null || true' INT TERM
wait "$HO_PID"
wait "$MF_PID"
"$PY" scripts/286_generate_resubmission7_holdout_eval.py --workers 18
"$PY" scripts/287_evaluate_resubmission7_holdout.py --threads 12
"$PY" scripts/290_audit_resubmission7_holdout.py
"$PY" scripts/289_report_resubmission7_holdout.py
