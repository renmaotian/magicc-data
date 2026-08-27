#!/bin/bash
# =============================================================================
# WS2 finisher — waits for the running CoCoPyE job on Set F, then:
#   1. parses CheckM2 / CoCoPyE / DeepCheck predictions (scripts/91)
#   2. runs the WS2.5 attribution analysis with every available tool
#      (scripts/147), which also re-runs the provenance audit and writes the
#      type x distance signed-error heatmaps.
#
# GUNC is deliberately not touched — Set F GUNC is owned by the WS4.2 agent and
# a long GUNC job is running concurrently.
#
# Safe to run detached: it tolerates CoCoPyE failing (the analysis then simply
# proceeds with the tools that did produce predictions).
#
# Usage: setsid nohup scripts/148_finish_ws2_set_F.sh > logs/revision/ws2_finish.log 2>&1 &
# =============================================================================
set -uo pipefail

PROJECT_DIR="/path/to/magicc"
SET_NAME="${1:-set_F}"
SET_DIR="${PROJECT_DIR}/data/benchmarks/${SET_NAME}"
LOG_DIR="${PROJECT_DIR}/logs/revision"
WORKERS=10

cd "${PROJECT_DIR}" || exit 1
mkdir -p "${LOG_DIR}"

echo "=============================================================="
echo "WS2 finisher for ${SET_NAME} -- $(date -Is)"
echo "=============================================================="

# ---------------------------------------------------------------- 1. wait
if pgrep -f "bin/cocopye run" > /dev/null; then
    echo "[wait] CoCoPyE is running; waiting for it to finish ..."
    while pgrep -f "bin/cocopye run" > /dev/null; do sleep 30; done
    echo "[wait] CoCoPyE process ended at $(date -Is)"
fi

if [[ -f "${SET_DIR}/cocopye_raw_output.csv" ]]; then
    N=$(( $(wc -l < "${SET_DIR}/cocopye_raw_output.csv") - 1 ))
    echo "[cocopye] output present: ${N} rows"
    TOOLS="checkm2,cocopye,deepcheck"
else
    echo "[cocopye] WARNING: no cocopye_raw_output.csv -- proceeding without CoCoPyE"
    tail -20 "${LOG_DIR}/cocopye_${SET_NAME}.log" 2>/dev/null
    TOOLS="checkm2,deepcheck"
fi

# ------------------------------------------------------- 2. parse predictions
echo
echo "[parse] scripts/91 --sets ${SET_NAME} --tools ${TOOLS}"
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 \
  conda run -n magicc2 python "${PROJECT_DIR}/scripts/91_parse_competitor_clean_cd.py" \
    --sets "${SET_NAME}" --tools "${TOOLS}" --torch-threads 4 2>&1 | tail -20

# ------------------------------------------------------------- 3. analysis
echo
echo "[analyse] scripts/147 --set ${SET_NAME} --workers ${WORKERS}"
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES="" \
  conda run -n magicc2 python "${PROJECT_DIR}/scripts/147_analyze_contamination_types.py" \
    --set "${SET_NAME}" --workers "${WORKERS}" 2>&1 \
  | tee "${LOG_DIR}/ws2_analysis_final.log" | tail -5

echo
echo "=============================================================="
echo "WS2 finisher done -- $(date -Is)"
ls -la "${PROJECT_DIR}/results/revision/${SET_NAME}/"
echo "=============================================================="
