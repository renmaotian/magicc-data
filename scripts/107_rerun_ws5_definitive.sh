#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 107_rerun_ws5_definitive.sh
#
# Definitive WS5 metrics re-run, now that CheckM2 / CoCoPyE / DeepCheck
# predictions exist for set_C_clean and set_D_clean.
#
# Collapse-safe and resumable: each stage writes a sentinel into
# results/revision/metrics/.stage_done/ on success and is skipped on re-run.
# Delete a sentinel (or pass --force) to redo a stage.
#
#   bash scripts/107_rerun_ws5_definitive.sh [--force]
# ---------------------------------------------------------------------------
set -u -o pipefail

ROOT=/path/to/magicc
PY=/path/to/conda/envs/magicc2/bin/python
OUT="$ROOT/results/revision/metrics"
SENT="$OUT/.stage_done"
LOG="$ROOT/logs/revision"

cd "$ROOT" || exit 1
mkdir -p "$SENT" "$LOG"

# Keep every numeric library single-threaded: a GPU training job and other
# agents are running concurrently and the budget is <= 6 workers.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=""      # never touch the GPU
# Belt and braces for reproducible bootstrap CIs. Scripts 102-105 now derive
# every seed through fw.stable_hash (CRC-32), so this is no longer load-bearing,
# but pinning it means nothing in the chain can depend on the salted builtin.
export PYTHONHASHSEED=0

if [ "${1:-}" = "--force" ]; then rm -f "$SENT"/*; fi

run_stage () {
    local name="$1"; shift
    if [ -f "$SENT/$name" ]; then
        echo "[skip] $name (sentinel present)"
        return 0
    fi
    echo "[run ] $name  ->  $LOG/ws5_rerun_$name.log   ($(date '+%F %T'))"
    if "$@" > "$LOG/ws5_rerun_$name.log" 2>&1; then
        date '+%F %T' > "$SENT/$name"
        echo "[ok  ] $name  ($(date '+%F %T'))"
    else
        echo "[FAIL] $name -- see $LOG/ws5_rerun_$name.log"
        tail -25 "$LOG/ws5_rerun_$name.log"
        return 1
    fi
}

run_stage selftest   "$PY" scripts/101_metrics_framework.py --selftest   || exit 1
run_stage inventory  "$PY" scripts/101_metrics_framework.py --inventory  || exit 1
run_stage 102_mimag_thresholds "$PY" scripts/102_mimag_and_thresholds.py || exit 1
run_stage 103_signed_errors    "$PY" scripts/103_signed_errors.py        || exit 1
run_stage 104_clustered_stats  "$PY" scripts/104_clustered_statistics.py || exit 1
run_stage 105_plots            "$PY" scripts/105_distribution_plots.py   || exit 1
run_stage 106_domain_size_gunc "$PY" scripts/106_domain_size_and_gunc.py --threads 6 || exit 1

echo "[done] all WS5 stages complete ($(date '+%F %T'))"
