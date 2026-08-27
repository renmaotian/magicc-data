#!/bin/bash
# WS8.1 — execute ONE timed cell of the matched-thread speed benchmark.
#
#   bash scripts/161_ws8_run_one.sh <tool> <threads> <repeat> <inputset> <cache>
#
#     tool     : magicc | checkm2 | cocopye | deepcheck | deepcheck_e2e
#     threads  : 1 | 8 | 16 | 32 | ...
#     repeat   : 1 | 2 | 3 ...
#     inputset : set_E_100 | set_E_full
#     cache    : warm | cold
#
# WALL-CLOCK DEFINITION (applied uniformly to every tool):
#   process start  ->  results file written & process exited,
#   INCLUDING interpreter start, imports, model/database load and output write.
#   Measured by /usr/bin/time -v on the tool's own top-level command
#   (no `conda run` wrapper: it adds a second interpreter and buffers output).
#
# Load average is recorded immediately before and immediately after each run.
# Idempotent/resumable: an existing, complete .json for the cell is not re-run.
set -uo pipefail

PROJECT="/path/to/magicc"
ENVS="/path/to/anaconda3/envs"
CHECKM2_DB="${PROJECT}/tools/checkm2_db/CheckM2_database/uniref100.KO.1.dmnd"
SPEED="${PROJECT}/results/revision/speed"
RUNS="${SPEED}/runs"
SCRATCH="${SPEED}/scratch"

TOOL="${1:?tool}"; THREADS="${2:?threads}"; REP="${3:?repeat}"
INPUTSET="${4:?inputset}"; CACHE="${5:-warm}"

CELL="${TOOL}__t${THREADS}__${INPUTSET}__${CACHE}__r${REP}"
JSON="${RUNS}/${CELL}.json"
TIMEF="${RUNS}/${CELL}.time.txt"
STDOUTF="${RUNS}/${CELL}.stdout.txt"

mkdir -p "${RUNS}" "${SCRATCH}"

if [ -s "${JSON}" ] && grep -q '"wall_clock_s"' "${JSON}" 2>/dev/null; then
    echo "[skip] ${CELL} already complete"
    exit 0
fi

INLIST="${SPEED}/inputs/${INPUTSET}.txt"
INDIR="${SPEED}/inputs/${INPUTSET}"
if [ "${INPUTSET}" = "set_E_full" ]; then
    INDIR="${PROJECT}/data/benchmarks/set_E/fasta"
fi
NGEN=$(wc -l < "${INLIST}")

WORK="${SCRATCH}/${CELL}"
rm -rf "${WORK}"; mkdir -p "${WORK}"

# ---- optional cold-cache: evict the page cache for the exact files this run
# ---- will read (inputs + model/DB). Requires no root; uses posix_fadvise
# ---- POSIX_FADV_DONTNEED, which drops clean pages of the named files only.
if [ "${CACHE}" = "cold" ]; then
    python3 "${PROJECT}/scripts/162_ws8_drop_file_cache.py" --tool "${TOOL}" --input-list "${INLIST}" \
        > "${RUNS}/${CELL}.evict.txt" 2>&1
elif [ "${CACHE}" = "cold_verified" ]; then
    # 2026-07-31: script 162 crashed (TypeError) before evicting anything, so the
    # original cache=cold cells were warm. 176 is the fixed evictor and writes a
    # machine-readable mincore(2) residency report proving the eviction happened.
    python3 "${PROJECT}/scripts/176_ws8_evict_fixed.py" --tool "${TOOL}" --input-list "${INLIST}" \
        --json-out "${RUNS}/${CELL}.evict.json" > "${RUNS}/${CELL}.evict.txt" 2>&1
fi

LOAD_BEFORE=$(cut -d' ' -f1-3 /proc/loadavg)
NPROC_BUSY_BEFORE=$(ps -eo pcpu= | awk '{s+=$1} END {printf "%.1f", s}')
T_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# ---------------------------------------------------------------- dispatch
case "${TOOL}" in

  magicc)
    # end-to-end CLI: process start -> predictions TSV written.
    # ONNX/OMP thread counts are pinned so `--threads N` really means N cores.
    OUTF="${WORK}/magicc_predictions.tsv"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /usr/bin/time -v -o "${TIMEF}" \
        "${ENVS}/magicc2/bin/magicc" predict \
            --input-list "${INLIST}" \
            --output "${OUTF}" \
            --threads "${THREADS}" \
            --model "${PROJECT}/models/magicc_v5.onnx" \
        > "${STDOUTF}" 2>&1
    RC=$?
    NOUT=$(( $(wc -l < "${OUTF}" 2>/dev/null || echo 1) - 1 ))
    ;;

  magicc_condarun)
    # IDENTICAL to `magicc` except that it is invoked through `conda run`, which is
    # how the archived 97.5 s figure was produced. Isolates the wrapper's cost as a
    # candidate explanation for the historical 74.4 s vs 97.5 s gap (WS8.3).
    OUTF="${WORK}/magicc_predictions.tsv"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /usr/bin/time -v -o "${TIMEF}" \
        conda run -n magicc2 magicc predict \
            --input-list "${INLIST}" \
            --output "${OUTF}" \
            --threads "${THREADS}" \
            --model "${PROJECT}/models/magicc_v5.onnx" \
        > "${STDOUTF}" 2>&1
    RC=$?
    NOUT=$(( $(wc -l < "${OUTF}" 2>/dev/null || echo 1) - 1 ))
    ;;

  checkm2)
    # end-to-end: Prodigal -> DIAMOND -> feature vectors -> ML -> quality_report.tsv
    /usr/bin/time -v -o "${TIMEF}" \
        env CHECKM2DB="${CHECKM2_DB}" \
            PATH="${ENVS}/checkm2_py39/bin:${PATH}" \
            OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
            TF_CPP_MIN_LOG_LEVEL=3 \
        "${ENVS}/checkm2_py39/bin/checkm2" predict \
            --input "${INDIR}" \
            --output-directory "${WORK}/checkm2_out" \
            --threads "${THREADS}" \
            --extension .fasta \
            --force --remove_intermediates \
        > "${STDOUTF}" 2>&1
    RC=$?
    NOUT=$(( $(wc -l < "${WORK}/checkm2_out/quality_report.tsv" 2>/dev/null || echo 1) - 1 ))
    ;;

  checkm2_vectors)
    # identical to `checkm2` but keeps the pickled feature vectors DeepCheck needs.
    /usr/bin/time -v -o "${TIMEF}" \
        env CHECKM2DB="${CHECKM2_DB}" \
            PATH="${ENVS}/checkm2_py39/bin:${PATH}" \
            OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
            TF_CPP_MIN_LOG_LEVEL=3 \
        "${ENVS}/checkm2_py39/bin/checkm2" predict \
            --input "${INDIR}" \
            --output-directory "${WORK}/checkm2_out" \
            --threads "${THREADS}" \
            --extension .fasta \
            --force --dbg_vectors \
        > "${STDOUTF}" 2>&1
    RC=$?
    NOUT=$(( $(wc -l < "${WORK}/checkm2_out/quality_report.tsv" 2>/dev/null || echo 1) - 1 ))
    ;;

  cocopye)
    OUTF="${WORK}/cocopye_output.csv"
    /usr/bin/time -v -o "${TIMEF}" \
        env OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
            PATH="${ENVS}/magicc2/bin:${PATH}" \
        "${ENVS}/magicc2/bin/cocopye" run \
            -i "${INDIR}" \
            -o "${OUTF}" \
            -t "${THREADS}" \
            -v full \
        > "${STDOUTF}" 2>&1
    RC=$?
    NOUT=$(( $(wc -l < "${OUTF}" 2>/dev/null || echo 1) - 1 ))
    ;;

  deepcheck)
    # inference only, from pre-computed CheckM2 feature vectors
    # (this is the configuration the submitted Table S4 reported).
    OUTF="${WORK}/deepcheck_predictions.tsv"
    /usr/bin/time -v -o "${TIMEF}" \
        env OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
        "${ENVS}/magicc2/bin/python" "${PROJECT}/scripts/163_ws8_deepcheck_infer.py" \
            --features "${SPEED}/deepcheck_features/${INPUTSET}" \
            --output "${OUTF}" \
            --threads "${THREADS}" \
        > "${STDOUTF}" 2>&1
    RC=$?
    NOUT=$(( $(wc -l < "${OUTF}" 2>/dev/null || echo 1) - 1 ))
    ;;

  *)
    echo "unknown tool: ${TOOL}" >&2; exit 2 ;;
esac
# --------------------------------------------------------------------------

T_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
LOAD_AFTER=$(cut -d' ' -f1-3 /proc/loadavg)

python3 "${PROJECT}/scripts/164_ws8_parse_time.py" \
    --time-file "${TIMEF}" \
    --json "${JSON}" \
    --tool "${TOOL}" --threads "${THREADS}" --repeat "${REP}" \
    --input-set "${INPUTSET}" --cache "${CACHE}" \
    --n-genomes "${NGEN}" --n-output "${NOUT}" \
    --rc "${RC}" \
    --t-start "${T_START}" --t-end "${T_END}" \
    --load-before "${LOAD_BEFORE}" --load-after "${LOAD_AFTER}" \
    --busy-before "${NPROC_BUSY_BEFORE}"

# keep outputs small: drop the big intermediate trees, keep the result table
if [ "${TOOL}" = "checkm2" ] || [ "${TOOL}" = "checkm2_vectors" ]; then
    if [ "${TOOL}" = "checkm2_vectors" ]; then
        mkdir -p "${SPEED}/deepcheck_features/${INPUTSET}"
        cp -f "${WORK}"/checkm2_out/*.pkl "${SPEED}/deepcheck_features/${INPUTSET}/" 2>/dev/null
    fi
    cp -f "${WORK}/checkm2_out/quality_report.tsv" "${RUNS}/${CELL}.result.tsv" 2>/dev/null
elif [ "${TOOL}" = "cocopye" ]; then
    cp -f "${WORK}/cocopye_output.csv" "${RUNS}/${CELL}.result.csv" 2>/dev/null
else
    cp -f "${WORK}"/*predictions.tsv "${RUNS}/${CELL}.result.tsv" 2>/dev/null
fi
rm -rf "${WORK}"

echo "[done] ${CELL} rc=${RC}"
exit 0
