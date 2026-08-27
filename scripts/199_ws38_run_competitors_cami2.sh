#!/bin/bash
# =============================================================================
# WS3.8 (R1-M3) -- competitor tools on the CAMI II bin sets.
#
# Protocol names 92/93 were already taken (92_clean_cd_metrics.py,
# 93_leakage_specificity_control.py), so WS3.8 uses 195-201.
#
# Tools:
#   CheckM2 1.0.1  (env checkm2_py39, --dbg_vectors so DeepCheck can reuse the
#                   feature vectors -- DeepCheck is a pure transform of those
#                   PKL vectors, not a separate external run)
#   CoCoPyE 0.5.0  (env magicc2)
#
# CheckM2 is the throughput bottleneck (~0.8 genomes/min/thread). Bin sets are
# therefore SUBSAMPLED DELIBERATELY where needed by
# scripts/200_ws38_cami2_select_competitor_cohort.py, which writes a symlink
# farm under data/real_data/cami2/bins/<ds>/<binset>_competitor_subset/ and a
# TSV recording exactly which bins were selected and why. Any subsampling is
# reported in the results.
#
# Resumable: a tool is skipped when its output already covers every input bin.
#
# Usage:
#   scripts/199_ws38_run_competitors_cami2.sh --dataset strain_madness \
#       --binset mixed [--threads 24] [--tools checkm2,cocopye]
# =============================================================================

set -uo pipefail

PROJECT_DIR="/path/to/magicc"
CHECKM2_DB="${PROJECT_DIR}/tools/checkm2_db/CheckM2_database/uniref100.KO.1.dmnd"
CHECKM2_ENV="checkm2_py39"
COCOPYE_ENV="magicc2"
LOG_DIR="${PROJECT_DIR}/logs/revision"
OUT_ROOT="${PROJECT_DIR}/results/revision/cami2/competitors"

THREADS=24
DATASET=""
BINSET=""
TOOLS="checkm2,cocopye"
SUBSET_SUFFIX="_competitor_subset"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --threads) THREADS="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --binset)  BINSET="$2";  shift 2 ;;
        --tools)   TOOLS="$2";   shift 2 ;;
        --full)    SUBSET_SUFFIX=""; shift 1 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -n "${DATASET}" && -n "${BINSET}" ]] || { echo "need --dataset and --binset" >&2; exit 2; }

FASTA_DIR="${PROJECT_DIR}/data/real_data/cami2/bins/${DATASET}/${BINSET}${SUBSET_SUFFIX}"
[[ -d "${FASTA_DIR}" ]] || { echo "ERROR: ${FASTA_DIR} missing" >&2; exit 1; }
N_GENOMES=$(find "${FASTA_DIR}/" -maxdepth 1 -name '*.fasta' | wc -l)
[[ "${N_GENOMES}" -gt 0 ]] || { echo "ERROR: no .fasta in ${FASTA_DIR}" >&2; exit 1; }

TAG="${DATASET}_${BINSET}"
WORK="${OUT_ROOT}/${TAG}"
mkdir -p "${WORK}" "${LOG_DIR}"

NOTE="ACCURACY RUN, not the controlled speed benchmark (WS8.1). Wall-clock is indicative only."
LOADAVG_AT_START=$(cut -d' ' -f1-3 /proc/loadavg | tr ' ' '/')

echo "=============================================================="
echo "WS3.8 competitors on CAMI II ${TAG} -- $(date -Is)"
echo "bins=${N_GENOMES}  threads=${THREADS}  tools=${TOOLS}"
echo "input=${FASTA_DIR}"
echo "load=$(cat /proc/loadavg)"
echo "=============================================================="

has_tool() { [[ ",${TOOLS}," == *",$1,"* ]]; }

write_wallclock() {
    { echo "tool=$2"; echo "version=$3"; echo "cohort=${TAG}"; echo "n_bins=${N_GENOMES}"
      echo "wall_clock_s=$4"; echo "threads=${THREADS}"; echo "started=$5"
      echo "finished=$6"; echo "host=$(hostname)"; echo "n_cpus_total=$(nproc)"
      echo "loadavg_at_start=${LOADAVG_AT_START}"; echo "command=$7"
      echo "note=${NOTE}"; } > "$1"
}

# ------------------------------------------------------------------ CheckM2
if has_tool checkm2; then
    OUT_DIR="${WORK}/checkm2_output"
    QR="${OUT_DIR}/quality_report.tsv"
    WC="${WORK}/checkm2_wallclock.txt"
    LOG="${LOG_DIR}/ws3.8_checkm2_${TAG}.log"
    N_DONE=0; [[ -f "${QR}" ]] && N_DONE=$(($(wc -l < "${QR}") - 1))
    if [[ "${N_DONE}" -ge "${N_GENOMES}" && -f "${WC}" ]]; then
        echo "[checkm2] SKIP -- ${N_DONE} rows already"
    else
        echo "[checkm2] running (${N_DONE}/${N_GENOMES}) ..."
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
            write_wallclock "${WC}" checkm2 "CheckM2 1.0.1" "${ELAPSED}" \
                "${STARTED}" "${FINISHED}" "${CMD}"
            echo "[checkm2] done ${ELAPSED}s -- ${N_DONE} rows"
        fi
    fi
fi

# ------------------------------------------------------------------ CoCoPyE
if has_tool cocopye; then
    RAW="${WORK}/cocopye_raw_output.csv"
    WC="${WORK}/cocopye_wallclock.txt"
    LOG="${LOG_DIR}/ws3.8_cocopye_${TAG}.log"
    N_DONE=0; [[ -f "${RAW}" ]] && N_DONE=$(($(wc -l < "${RAW}") - 1))
    if [[ "${N_DONE}" -ge "${N_GENOMES}" && -f "${WC}" ]]; then
        echo "[cocopye] SKIP -- ${N_DONE} rows already"
    else
        echo "[cocopye] running (${N_DONE}/${N_GENOMES}) ..."
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

echo "WS3.8 competitors on ${TAG} FINISHED -- $(date -Is)"
