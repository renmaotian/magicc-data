#!/bin/bash
# =============================================================================
# WS1.11 (R1-m13) -- run CheckM2, CoCoPyE and DeepCheck on set_H_ncbi.
#
# Invocations are reused VERBATIM from scripts/090_run_competitors_clean_cd.sh so the
# competitor numbers on set_H_ncbi are directly comparable with those on
# set_C_clean / set_D_clean / set_F / set_G:
#   CheckM2 1.0.1 -- env checkm2_py39, CHECKM2DB, --dbg_vectors (DeepCheck consumes
#                    those feature vectors)
#   CoCoPyE 0.5.0 -- env magicc2, cocopye run -i <dir> -o <csv> -t <n> -v full
#   DeepCheck     -- executed inside scripts/091_parse_competitor_clean_cd.py, which is
#                    a pure tensor transform of CheckM2's --dbg_vectors PKLs
#
# GUNC is deliberately NOT run: it is a detection comparator (CSS + pass/fail, no
# completeness/contamination percentages), so it cannot answer R1-m13's question about
# estimate error. Recorded as a scope decision, not an omission.
#
# CheckM2's own error on the H_fail arm is one of the headline results of WS1.11: if
# CheckM2 is wrong on the genomes its own scores excluded from the reference pool, that
# is precisely the blind spot the original curation created.
#
# Resumable / collapse-safe: each (tool, set) pair is skipped when already complete.
#
# Usage: scripts/190_ws1_11_run_competitors.sh [--threads N] [--tools a,b,c]
# =============================================================================

set -uo pipefail

PROJECT_DIR="/path/to/magicc"
CHECKM2_DB="${PROJECT_DIR}/tools/checkm2_db/CheckM2_database/uniref100.KO.1.dmnd"
CHECKM2_ENV="checkm2_py39"
COCOPYE_ENV="magicc2"
BENCHMARK_DIR="${PROJECT_DIR}/data/benchmarks"
LOG_DIR="${PROJECT_DIR}/logs/revision"
SET_NAME="set_H_ncbi"

THREADS=24
TOOLS="checkm2,cocopye,deepcheck"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --threads) THREADS="$2"; shift 2 ;;
        --tools)   TOOLS="$2";   shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "${LOG_DIR}"
NOTE="ACCURACY RUN, not the controlled speed benchmark (WS8.1). Wall-clock is indicative only."
has_tool() { [[ ",${TOOLS}," == *",$1,"* ]]; }

write_wallclock() {
    local f="$1" tool="$2" setname="$3" version="$4" elapsed="$5" started="$6" finished="$7" cmd="$8"
    {
        echo "tool=${tool}"; echo "version=${version}"; echo "set=${setname}"
        echo "wall_clock_s=${elapsed}"; echo "threads=${THREADS}"
        echo "started=${started}"; echo "finished=${finished}"
        echo "host=$(hostname)"; echo "n_cpus_total=$(nproc)"
        echo "loadavg_at_start=${LOADAVG_AT_START}"
        echo "command=${cmd}"; echo "note=${NOTE}"
    } > "${f}"
}

SET_DIR="${BENCHMARK_DIR}/${SET_NAME}"
FASTA_DIR="${SET_DIR}/fasta"
[[ -d "${FASTA_DIR}" ]] || { echo "ERROR: ${FASTA_DIR} does not exist" >&2; exit 1; }
N_GENOMES=$(find "${FASTA_DIR}" -maxdepth 1 -name '*.fasta' | wc -l)

echo "=============================================================="
echo "WS1.11 competitor runs on ${SET_NAME} -- $(date -Is)"
echo "  genomes: ${N_GENOMES}   threads: ${THREADS}   tools: ${TOOLS}"
echo "  load   : $(cat /proc/loadavg)"
echo "=============================================================="

# ------------------------------------------------------------------- CheckM2
if has_tool checkm2; then
    OUT_DIR="${SET_DIR}/checkm2_output"
    QR="${OUT_DIR}/quality_report.tsv"
    WC_FILE="${SET_DIR}/checkm2_wallclock.txt"
    LOG_FILE="${LOG_DIR}/ws1.11_checkm2_${SET_NAME}.log"
    N_DONE=0
    [[ -f "${QR}" ]] && N_DONE=$(($(wc -l < "${QR}") - 1))
    N_PKL=$(find "${OUT_DIR}" -maxdepth 1 -name '*.pkl' 2>/dev/null | wc -l)
    if [[ "${N_DONE}" -ge "${N_GENOMES}" && "${N_PKL}" -ge 1 && -f "${WC_FILE}" ]]; then
        echo "[checkm2] SKIP -- ${N_DONE} rows, ${N_PKL} PKL(s), wallclock recorded"
    else
        echo "[checkm2] running (${N_DONE}/${N_GENOMES} rows present) ..."
        LOADAVG_AT_START=$(cut -d' ' -f1-3 /proc/loadavg | tr ' ' '/')
        STARTED=$(date -Is); T0=$(date +%s)
        CMD="conda run -n ${CHECKM2_ENV} env CHECKM2DB=${CHECKM2_DB} checkm2 predict --threads ${THREADS} -x .fasta --input ${FASTA_DIR} --output-directory ${OUT_DIR} --force --dbg_vectors"
        conda run -n "${CHECKM2_ENV}" env CHECKM2DB="${CHECKM2_DB}" \
            checkm2 predict --threads "${THREADS}" -x .fasta \
            --input "${FASTA_DIR}" --output-directory "${OUT_DIR}" \
            --force --dbg_vectors > "${LOG_FILE}" 2>&1
        RC=$?
        T1=$(date +%s); ELAPSED=$((T1 - T0)); FINISHED=$(date -Is)
        if [[ ${RC} -ne 0 ]]; then
            echo "[checkm2] FAILED rc=${RC}; tail of ${LOG_FILE}:" >&2
            tail -40 "${LOG_FILE}" >&2
        else
            N_DONE=$(($(wc -l < "${QR}") - 1))
            N_PKL=$(find "${OUT_DIR}" -maxdepth 1 -name '*.pkl' | wc -l)
            write_wallclock "${WC_FILE}" checkm2 "${SET_NAME}" "CheckM2 1.0.1" \
                "${ELAPSED}" "${STARTED}" "${FINISHED}" "${CMD}"
            echo "[checkm2] done in ${ELAPSED}s -- ${N_DONE} rows, ${N_PKL} PKL(s)"
        fi
    fi
fi

# ------------------------------------------------------------------- CoCoPyE
if has_tool cocopye; then
    RAW="${SET_DIR}/cocopye_raw_output.csv"
    WC_FILE="${SET_DIR}/cocopye_wallclock.txt"
    LOG_FILE="${LOG_DIR}/ws1.11_cocopye_${SET_NAME}.log"
    N_DONE=0
    [[ -f "${RAW}" ]] && N_DONE=$(($(wc -l < "${RAW}") - 1))
    if [[ "${N_DONE}" -ge "${N_GENOMES}" && -f "${WC_FILE}" ]]; then
        echo "[cocopye] SKIP -- ${N_DONE} rows present"
    else
        echo "[cocopye] running (${N_DONE}/${N_GENOMES} rows present) ..."
        LOADAVG_AT_START=$(cut -d' ' -f1-3 /proc/loadavg | tr ' ' '/')
        STARTED=$(date -Is); T0=$(date +%s)
        CMD="conda run -n ${COCOPYE_ENV} cocopye run -i ${FASTA_DIR} -o ${RAW} -t ${THREADS} -v full"
        conda run -n "${COCOPYE_ENV}" cocopye run -i "${FASTA_DIR}" -o "${RAW}" \
            -t "${THREADS}" -v full > "${LOG_FILE}" 2>&1
        RC=$?
        T1=$(date +%s); ELAPSED=$((T1 - T0)); FINISHED=$(date -Is)
        if [[ ${RC} -ne 0 || ! -f "${RAW}" ]]; then
            echo "[cocopye] FAILED rc=${RC}; tail of ${LOG_FILE}:" >&2
            tail -40 "${LOG_FILE}" >&2
        else
            N_DONE=$(($(wc -l < "${RAW}") - 1))
            write_wallclock "${WC_FILE}" cocopye "${SET_NAME}" "CoCoPyE 0.5.0" \
                "${ELAPSED}" "${STARTED}" "${FINISHED}" "${CMD}"
            echo "[cocopye] done in ${ELAPSED}s -- ${N_DONE} rows"
        fi
    fi
fi

# ---------------------------------------- parse / merge / DeepCheck inference
echo ""
echo "[parse] scripts/091_parse_competitor_clean_cd.py --sets ${SET_NAME}"
VERIF="${PROJECT_DIR}/results/revision/benchmark/clean_cd_merge_verification.json"
BACKUP=""
if [[ -f "${VERIF}" ]]; then
    BACKUP="${VERIF}.ws1.11_backup"
    cp -p "${VERIF}" "${BACKUP}"
fi
PARSE_TOOLS=$(echo "${TOOLS}" | tr ',' '\n' | grep -v '^$' | paste -sd, -)
conda run -n "${COCOPYE_ENV}" env PYTHONHASHSEED=0 python \
    "${PROJECT_DIR}/scripts/091_parse_competitor_clean_cd.py" \
    --sets "${SET_NAME}" --tools "${PARSE_TOOLS}" \
    > "${LOG_DIR}/ws1.11_parse_${SET_NAME}.log" 2>&1
RC=$?
tail -30 "${LOG_DIR}/ws1.11_parse_${SET_NAME}.log"
if [[ -f "${VERIF}" ]]; then
    mkdir -p "${PROJECT_DIR}/results/revision/circularity"
    cp -p "${VERIF}" "${PROJECT_DIR}/results/revision/circularity/ws1_11_merge_verification.json"
fi
[[ -n "${BACKUP}" ]] && mv -f "${BACKUP}" "${VERIF}"
echo "[parse] rc=${RC}"

echo ""
echo "=============================================================="
echo "WS1.11 competitor runs finished -- $(date -Is)"
for f in checkm2_predictions.tsv cocopye_predictions.tsv deepcheck_predictions.tsv \
         magicc_v5_predictions.tsv; do
    if [[ -f "${SET_DIR}/${f}" ]]; then
        echo "  ${f}: $(($(wc -l < "${SET_DIR}/${f}") - 1)) rows"
    else
        echo "  ${f}: MISSING"
    fi
done
echo "=============================================================="
