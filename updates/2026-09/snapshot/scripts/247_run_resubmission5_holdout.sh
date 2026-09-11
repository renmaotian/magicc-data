#!/usr/bin/env bash
# Resume-safe full experiment: two matched models, identical complete V5 recipe.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export NUMBA_NUM_THREADS=1
export PYTHONHASHSEED=0
PY=${R5_PYTHON:-python}
R5_OUT=results/revision/holdout_resubmission5
mkdir -p "$R5_OUT/logs"
run_arm() {
    local variant=$1
    "$PY" scripts/243_build_resubmission5_holdout_data.py --variant "$variant" --workers 12
    "$PY" scripts/244_train_resubmission5_holdout.py --variant "$variant" --threads 12
}
run_arm holdout > "$R5_OUT/logs/holdout_full_pipeline.log" 2>&1 &
HO_PID=$!
run_arm matched_full > "$R5_OUT/logs/matched_full_pipeline.log" 2>&1 &
MF_PID=$!
trap 'kill "$HO_PID" "$MF_PID" 2>/dev/null || true' INT TERM
wait "$HO_PID"
wait "$MF_PID"
"$PY" scripts/245_generate_resubmission5_holdout_eval.py --workers 8
"$PY" scripts/246_evaluate_resubmission5_holdout.py
"$PY" scripts/248_report_resubmission5_holdout.py
