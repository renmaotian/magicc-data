#!/bin/bash
# =============================================================================
# WS6.3 -- run the competitor tools on Set G (sequencing / assembly error
# robustness).  Protocol name: part of 109_error_robustness_analysis.py;
# split out as a shell runner following scripts/146_run_tools_set_F.sh.
#
# Tools run here:
#   CheckM2 1.0.1  (env checkm2_py39, --dbg_vectors so DeepCheck gets its
#                   feature vectors)
#   CoCoPyE 0.5.0  (env magicc2)
# Then scripts/91_parse_competitor_clean_cd.py --sets set_G --tools
#   checkm2,cocopye,deepcheck performs the DeepCheck tensor transform and the
#   verified merges (DeepCheck is a pure transform of CheckM2's PKL vectors, so
#   it is not a separate external run).
#
# MAGICC V5 is run by scripts/153_error_robustness_analysis.py (direct ONNX
# against the frozen models/magicc_v5.onnx).
# GUNC is out of scope (it flags rather than quantifies, and WS4.2 owns it).
#
# THREADS ARE CAPPED (default 10): a GPU retrain, GUNC/diamond batches and other
# agents share this 48-core machine.  These are ACCURACY runs, not the WS8.1
# speed benchmark; wall-clock is indicative only and that caveat is written into
# every *_wallclock.txt.
#
# Resumable: each tool is skipped when its output is already complete.
#
# Usage:
#   scripts/152_run_tools_set_G.sh [--threads N] [--set set_G] [--tools ...]
# =============================================================================

set -uo pipefail

PROJECT_DIR="/path/to/magicc"
CHECKM2_DB="${PROJECT_DIR}/tools/checkm2_db/CheckM2_database/uniref100.KO.1.dmnd"
CHECKM2_ENV="checkm2_py39"
COCOPYE_ENV="magicc2"
BENCHMARK_DIR="${PROJECT_DIR}/data/benchmarks"
LOG_DIR="${PROJECT_DIR}/logs/revision"

THREADS=10
SET_NAME="set_G"
TOOLS="checkm2,cocopye,parse"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --threads) THREADS="$2"; shift 2 ;;
        --set)     SET_NAME="$2"; shift 2 ;;
        --tools)   TOOLS="$2";   shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "${LOG_DIR}"
SET_DIR="${BENCHMARK_DIR}/${SET_NAME}"
FASTA_DIR="${SET_DIR}/fasta"
[[ -d "${FASTA_DIR}" ]] || { echo "ERROR: ${FASTA_DIR} missing" >&2; exit 1; }
N_GENOMES=$(find "${FASTA_DIR}" -maxdepth 1 -name '*.fasta' | wc -l)

NOTE="ACCURACY RUN, not the controlled speed benchmark (WS8.1). Executed on a shared, concurrently loaded 48-core machine (other agents active), so wall-clock is indicative only."
LOADAVG_AT_START=$(cut -d' ' -f1-3 /proc/loadavg | tr ' ' '/')

echo "=============================================================="
echo "WS6.3 competitor runs on ${SET_NAME} -- $(date -Is)"
echo "genomes=${N_GENOMES}  threads=${THREADS}  tools=${TOOLS}"
echo "load=$(cat /proc/loadavg)"
echo "=============================================================="

has_tool() { [[ ",${TOOLS}," == *",$1,"* ]]; }

write_wallclock() {
    local f="$1" tool="$2" version="$3" elapsed="$4" started="$5" finished="$6" cmd="$7"
    {
        echo "tool=${tool}"; echo "version=${version}"; echo "set=${SET_NAME}"
        echo "wall_clock_s=${elapsed}"; echo "threads=${THREADS}"
        echo "started=${started}"; echo "finished=${finished}"
        echo "host=$(hostname)"; echo "n_cpus_total=$(nproc)"
        echo "loadavg_at_start=${LOADAVG_AT_START}"
        echo "command=${cmd}"; echo "note=${NOTE}"
    } > "${f}"
}

# ------------------------------------------------------------------ CheckM2
if has_tool checkm2; then
    OUT_DIR="${SET_DIR}/checkm2_output"
    QR="${OUT_DIR}/quality_report.tsv"
    WC="${SET_DIR}/checkm2_wallclock.txt"
    LOG="${LOG_DIR}/ws6_checkm2_${SET_NAME}.log"
    N_DONE=0; [[ -f "${QR}" ]] && N_DONE=$(($(wc -l < "${QR}") - 1))
    N_PKL=$(find "${OUT_DIR}" -maxdepth 1 -name '*.pkl' 2>/dev/null | wc -l)
    if [[ "${N_DONE}" -ge "${N_GENOMES}" && "${N_PKL}" -ge 1 && -f "${WC}" ]]; then
        echo "[checkm2] SKIP -- ${N_DONE} rows, ${N_PKL} PKL"
    else
        echo "[checkm2] running (${N_DONE}/${N_GENOMES}) ..."
        LOADAVG_AT_START=$(cut -d' ' -f1-3 /proc/loadavg | tr ' ' '/')
        STARTED=$(date -Is); T0=$(date +%s)
        CMD="conda run -n ${CHECKM2_ENV} env CHECKM2DB=${CHECKM2_DB} checkm2 predict --threads ${THREADS} -x .fasta --input ${FASTA_DIR} --output-directory ${OUT_DIR} --force --dbg_vectors"
        conda run -n "${CHECKM2_ENV}" env CHECKM2DB="${CHECKM2_DB}" \
            checkm2 predict --threads "${THREADS}" -x .fasta \
            --input "${FASTA_DIR}" --output-directory "${OUT_DIR}" \
            --force --dbg_vectors > "${LOG}" 2>&1
        RC=$?; T1=$(date +%s); ELAPSED=$((T1 - T0)); FINISHED=$(date -Is)
        if [[ ${RC} -ne 0 ]]; then
            echo "[checkm2] FAILED rc=${RC}" >&2; tail -30 "${LOG}" >&2
        else
            N_DONE=$(($(wc -l < "${QR}") - 1))
            N_PKL=$(find "${OUT_DIR}" -maxdepth 1 -name '*.pkl' | wc -l)
            write_wallclock "${WC}" checkm2 "CheckM2 1.0.1" "${ELAPSED}" \
                "${STARTED}" "${FINISHED}" "${CMD}"
            echo "[checkm2] done ${ELAPSED}s -- ${N_DONE} rows, ${N_PKL} PKL"
        fi
    fi
fi

# ------------------------------------------------------------------ CoCoPyE
if has_tool cocopye; then
    RAW="${SET_DIR}/cocopye_raw_output.csv"
    WC="${SET_DIR}/cocopye_wallclock.txt"
    LOG="${LOG_DIR}/ws6_cocopye_${SET_NAME}.log"
    N_DONE=0; [[ -f "${RAW}" ]] && N_DONE=$(($(wc -l < "${RAW}") - 1))
    if [[ "${N_DONE}" -ge "${N_GENOMES}" && -f "${WC}" ]]; then
        echo "[cocopye] SKIP -- ${N_DONE} rows"
    else
        echo "[cocopye] running (${N_DONE}/${N_GENOMES}) ..."
        LOADAVG_AT_START=$(cut -d' ' -f1-3 /proc/loadavg | tr ' ' '/')
        STARTED=$(date -Is); T0=$(date +%s)
        CMD="conda run -n ${COCOPYE_ENV} cocopye run -i ${FASTA_DIR} -o ${RAW} -t ${THREADS} -v full"
        conda run -n "${COCOPYE_ENV}" cocopye run -i "${FASTA_DIR}" -o "${RAW}" \
            -t "${THREADS}" -v full > "${LOG}" 2>&1
        RC=$?; T1=$(date +%s); ELAPSED=$((T1 - T0)); FINISHED=$(date -Is)
        if [[ ${RC} -ne 0 || ! -f "${RAW}" ]]; then
            echo "[cocopye] FAILED rc=${RC}" >&2; tail -30 "${LOG}" >&2
        else
            N_DONE=$(($(wc -l < "${RAW}") - 1))
            write_wallclock "${WC}" cocopye "CoCoPyE 0.5.0" "${ELAPSED}" \
                "${STARTED}" "${FINISHED}" "${CMD}"
            echo "[cocopye] done ${ELAPSED}s -- ${N_DONE} rows"
        fi
    fi
fi

# ------------------------------- DeepCheck + verified merges (script 91) ----
if has_tool parse; then
    echo "[parse] DeepCheck transform + verified merges via scripts/91 ..."
    conda run -n magicc2 python "${PROJECT_DIR}/scripts/91_parse_competitor_clean_cd.py" \
        --sets "${SET_NAME}" --tools checkm2,cocopye,deepcheck \
        --torch-threads 4 2>&1 | tail -30
fi

echo "=============================================================="
echo "WS6.3 tool runs finished -- $(date -Is)"
ls -la "${SET_DIR}"/*_predictions.tsv 2>/dev/null
