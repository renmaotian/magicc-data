#!/bin/bash
# =============================================================================
# WS3.8 (R1-M3) -- full CAMI II pipeline, definitive run.
#
# Stages
#   1  truth + bin sets            196   (both datasets, all samples on disk)
#   2  leakage / provenance audit  197
#   3  MAGICC V5                   198
#   4  competitor cohort selection 200 --select
#   5  CheckM2 + CoCoPyE           199   (two chains in parallel, 24 threads each)
#   6  collect all tools           200 --collect   (adds DeepCheck from CheckM2 PKLs)
#   7  analysis                    201
#   8  figures + report            202
#
# Collapse-safe: every stage is resumable and skips completed work. Launch under
# setsid/nohup so it survives the session.
#
# Usage:
#   setsid nohup scripts/203_ws38_cami2_pipeline.sh > logs/revision/ws3.8_pipeline.log 2>&1 &
# =============================================================================

set -uo pipefail

PROJECT_DIR="/path/to/magicc"
cd "${PROJECT_DIR}" || exit 1
source /path/to/anaconda3/etc/profile.d/conda.sh
conda activate magicc2
export PYTHONHASHSEED=0

LOG_DIR="${PROJECT_DIR}/logs/revision"
mkdir -p "${LOG_DIR}"
DATASETS="marine strain_madness"
THREADS=24

say() { echo "[$(date -Is)] $*"; }

say "=== WS3.8 CAMI II pipeline START ==="
say "samples on disk: marine=$(ls -d data/real_data/cami2/marine/sample_* 2>/dev/null | wc -l) strain=$(ls -d data/real_data/cami2/strain_madness/sample_* 2>/dev/null | wc -l)"

# --------------------------------------------------- 1. truth + bins
for DS in ${DATASETS}; do
    say "STAGE 1: truth + bins for ${DS}"
    python scripts/196_ws38_cami2_truth_and_bins.py --dataset "${DS}" \
        2>&1 | sed "s/^/  [196:${DS}] /"
done

# --------------------------------------------------- 2. leakage audit
for DS in ${DATASETS}; do
    say "STAGE 2: leakage audit for ${DS}"
    python scripts/197_ws38_cami2_leakage_audit.py --dataset "${DS}" \
        2>&1 | sed "s/^/  [197:${DS}] /"
done

# --------------------------------------------------- 3. MAGICC V5
for DS in ${DATASETS}; do
    for BS in gold mixed; do
        say "STAGE 3: MAGICC V5 on ${DS}/${BS}"
        python scripts/198_ws38_run_magicc_cami2.py --dataset "${DS}" --binset "${BS}" \
            --workers "${THREADS}" 2>&1 | sed "s/^/  [198:${DS}:${BS}] /"
    done
done

# --------------------------------------------------- 4. competitor cohorts
for DS in ${DATASETS}; do
    say "STAGE 4: competitor cohort selection for ${DS}"
    python scripts/200_ws38_cami2_cohort_and_collect.py --dataset "${DS}" \
        --binsets gold,mixed --select 2>&1 | sed "s/^/  [200sel:${DS}] /"
done

# --------------------------------------------------- 5. CheckM2 || CoCoPyE
say "STAGE 5: CheckM2 and CoCoPyE (parallel chains, ${THREADS} threads each)"
(
  for DS in ${DATASETS}; do
      for BS in mixed gold; do
          bash scripts/199_ws38_run_competitors_cami2.sh --dataset "${DS}" \
              --binset "${BS}" --threads "${THREADS}" --tools checkm2
      done
  done
) > "${LOG_DIR}/ws3.8_chain_checkm2.log" 2>&1 &
PID_CK=$!
(
  for DS in ${DATASETS}; do
      for BS in mixed gold; do
          bash scripts/199_ws38_run_competitors_cami2.sh --dataset "${DS}" \
              --binset "${BS}" --threads "${THREADS}" --tools cocopye
      done
  done
) > "${LOG_DIR}/ws3.8_chain_cocopye.log" 2>&1 &
PID_CO=$!
say "  checkm2 chain pid=${PID_CK}, cocopye chain pid=${PID_CO}"
wait ${PID_CK}; say "  checkm2 chain exited rc=$?"
wait ${PID_CO}; say "  cocopye chain exited rc=$?"

# --------------------------------------------------- 6. collect
for DS in ${DATASETS}; do
    say "STAGE 6: collect predictions for ${DS}"
    python scripts/200_ws38_cami2_cohort_and_collect.py --dataset "${DS}" \
        --binsets gold,mixed --collect 2>&1 | sed "s/^/  [200col:${DS}] /"
done

# --------------------------------------------------- 7-8. analysis + report
say "STAGE 7: analysis"
python scripts/201_ws38_cami2_analysis.py 2>&1 | sed 's/^/  [201] /'
say "STAGE 8: figures + report"
python scripts/202_ws38_cami2_figures_and_report.py 2>&1 | sed 's/^/  [202] /'

say "=== WS3.8 CAMI II pipeline DONE ==="
