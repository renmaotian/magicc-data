#!/bin/bash
# =============================================================================
# WS3 Track A — run CheckM2 and CoCoPyE on a directory of real-data genomes.
#
# Invocations are reused verbatim from scripts/90_run_competitors_clean_cd.sh so
# the real-data numbers are directly comparable with the synthetic benchmarks:
#   CheckM2 1.0.1  (env checkm2_py39, CHECKM2DB, --dbg_vectors so that DeepCheck's
#                   feature vectors are produced; DeepCheck itself is a pure tensor
#                   transform executed later by scripts/142)
#   CoCoPyE 0.5.0  (env magicc2, cocopye run -i DIR -o CSV -t N -v full)
#
# THREADS ARE CAPPED AT 12 by default: three other agents and a GPU training job
# share this 48-core machine (task constraint: <= 14 workers).
#
# Resumable: a tool is skipped when its output already covers every input genome.
#
# Usage:
#   scripts/141_run_competitors_realdata.sh --input DIR --output DIR \
#        [--ext .fasta] [--threads 12] [--tools checkm2,cocopye]
# =============================================================================
set -uo pipefail

PROJECT_DIR="/path/to/magicc"
CHECKM2_DB="${PROJECT_DIR}/tools/checkm2_db/CheckM2_database/uniref100.KO.1.dmnd"
LOG_DIR="${PROJECT_DIR}/logs/revision"

INPUT=""; OUTPUT=""; EXT=".fasta"; THREADS=12; TOOLS="checkm2,cocopye"; LABEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)   INPUT="$2";   shift 2 ;;
        --output)  OUTPUT="$2";  shift 2 ;;
        --ext)     EXT="$2";     shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --tools)   TOOLS="$2";   shift 2 ;;
        --label)   LABEL="$2";   shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -z "${INPUT}" || -z "${OUTPUT}" ]] && { echo "need --input and --output" >&2; exit 2; }
[[ -z "${LABEL}" ]] && LABEL=$(basename "${OUTPUT}")
mkdir -p "${OUTPUT}" "${LOG_DIR}"

N=$(find "${INPUT}" -maxdepth 1 -name "*${EXT}" | wc -l)
echo "=============================================================="
echo "WS3 Track A competitors — ${LABEL}"
echo "input=${INPUT} (${N} genomes, ext=${EXT})  threads=${THREADS}"
echo "load=$(cat /proc/loadavg)   started=$(date -Is)"
echo "=============================================================="
has_tool() { [[ ",${TOOLS}," == *",$1,"* ]]; }

NOTE="ACCURACY RUN, not the controlled speed benchmark (WS8.1). Shared, concurrently loaded 48-core machine; wall-clock indicative only."

# ------------------------------------------------------------------- CheckM2
if has_tool checkm2; then
    CK="${OUTPUT}/checkm2_output"; QR="${CK}/quality_report.tsv"
    NDONE=0; [[ -f "${QR}" ]] && NDONE=$(($(wc -l < "${QR}") - 1))
    if [[ "${NDONE}" -ge "${N}" ]]; then
        echo "[checkm2] SKIP — ${NDONE} rows present"
    else
        echo "[checkm2] running ..."
        T0=$(date +%s); STARTED=$(date -Is)
        conda run -n checkm2_py39 env CHECKM2DB="${CHECKM2_DB}" \
            checkm2 predict --threads "${THREADS}" -x "${EXT}" \
            --input "${INPUT}" --output-directory "${CK}" --force --dbg_vectors \
            > "${LOG_DIR}/checkm2_${LABEL}.log" 2>&1
        RC=$?; T1=$(date +%s)
        if [[ ${RC} -ne 0 ]]; then
            echo "[checkm2] FAILED rc=${RC}"; tail -30 "${LOG_DIR}/checkm2_${LABEL}.log" >&2
        else
            NDONE=$(($(wc -l < "${QR}") - 1))
            printf 'tool=checkm2\nversion=CheckM2 1.0.1\nset=%s\nwall_clock_s=%s\nthreads=%s\nstarted=%s\nfinished=%s\nn_genomes=%s\nnote=%s\n' \
                "${LABEL}" "$((T1-T0))" "${THREADS}" "${STARTED}" "$(date -Is)" "${NDONE}" "${NOTE}" \
                > "${OUTPUT}/checkm2_wallclock.txt"
            echo "[checkm2] done in $((T1-T0))s — ${NDONE} rows, $(find "${CK}" -maxdepth 1 -name '*.pkl' | wc -l) PKL"
        fi
    fi
fi

# ------------------------------------------------------------------- CoCoPyE
if has_tool cocopye; then
    RAW="${OUTPUT}/cocopye_raw_output.csv"
    NDONE=0; [[ -f "${RAW}" ]] && NDONE=$(($(wc -l < "${RAW}") - 1))
    if [[ "${NDONE}" -ge "${N}" ]]; then
        echo "[cocopye] SKIP — ${NDONE} rows present"
    else
        echo "[cocopye] running ..."
        T0=$(date +%s); STARTED=$(date -Is)
        conda run -n magicc2 cocopye run -i "${INPUT}" -o "${RAW}" \
            -t "${THREADS}" -v full > "${LOG_DIR}/cocopye_${LABEL}.log" 2>&1
        RC=$?; T1=$(date +%s)
        if [[ ${RC} -ne 0 || ! -f "${RAW}" ]]; then
            echo "[cocopye] FAILED rc=${RC}"; tail -30 "${LOG_DIR}/cocopye_${LABEL}.log" >&2
        else
            NDONE=$(($(wc -l < "${RAW}") - 1))
            printf 'tool=cocopye\nversion=CoCoPyE 0.5.0\nset=%s\nwall_clock_s=%s\nthreads=%s\nstarted=%s\nfinished=%s\nn_genomes=%s\nnote=%s\n' \
                "${LABEL}" "$((T1-T0))" "${THREADS}" "${STARTED}" "$(date -Is)" "${NDONE}" "${NOTE}" \
                > "${OUTPUT}/cocopye_wallclock.txt"
            echo "[cocopye] done in $((T1-T0))s — ${NDONE} rows"
        fi
    fi
fi
echo "finished $(date -Is)"
