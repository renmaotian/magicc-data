#!/bin/bash
# =============================================================================
# WS1.5 (competitor part) -- run CheckM2, CoCoPyE and GUNC on the clean
# replacement benchmark sets set_C_clean (Patescibacteriota) and set_D_clean
# (Archaea), 1,000 genomes each, built from 100 strictly held-out test-split
# references x 10 simulations.
#
# DeepCheck is NOT run here: it consumes CheckM2's intermediate feature vectors
# (written by --dbg_vectors) and its "run" is a pure tensor transform, so it is
# executed inside scripts/091_parse_competitor_clean_cd.py together with the
# merging/verification step.
#
# Invocations are reused verbatim from the established scripts:
#   CheckM2 1.0.1  -- scripts/036_run_checkm2_v2.sh   (env checkm2_py39, CHECKM2DB,
#                     --dbg_vectors so DeepCheck features are produced)
#   CoCoPyE 0.5.0  -- scripts/037_run_cocopye_v2.py   (env magicc2,
#                     cocopye run -i <dir> -o <csv> -t <n> -v full)
#   GUNC 1.1.1     -- scripts/076_run_gunc.py         (env gunc_env, DB tools/gunc_db)
#
# THREADS ARE CAPPED AT 16 by default: other agents share this machine
# (a GPU training job with ~24 CPU workers, plus a GUNC control run).
#
# These are ACCURACY runs, not the controlled speed benchmark (WS8.1). The
# machine is concurrently loaded, so the recorded wall-clock times are
# indicative only. That caveat is written into every *_wallclock.txt.
#
# Resumable / collapse-safe: every (tool, set) pair is skipped when its output
# is already complete, so the script can be re-run any number of times.
#
# Usage:
#   scripts/090_run_competitors_clean_cd.sh [--threads N]
#                                          [--tools checkm2,cocopye,gunc]
#                                          [--sets set_C_clean,set_D_clean]
# =============================================================================

set -uo pipefail

PROJECT_DIR="/path/to/magicc"
CHECKM2_DB="${PROJECT_DIR}/tools/checkm2_db/CheckM2_database/uniref100.KO.1.dmnd"
CHECKM2_ENV="checkm2_py39"
COCOPYE_ENV="magicc2"
GUNC_DB_DIR="${PROJECT_DIR}/tools/gunc_db"
BENCHMARK_DIR="${PROJECT_DIR}/data/benchmarks"
LOG_DIR="${PROJECT_DIR}/logs/revision"
GUNC_OUT_ROOT="${PROJECT_DIR}/results/revision/benchmark/gunc"

THREADS=16
TOOLS="checkm2,cocopye,gunc"
SETS="set_C_clean,set_D_clean"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --threads) THREADS="$2"; shift 2 ;;
        --tools)   TOOLS="$2";   shift 2 ;;
        --sets)    SETS="$2";    shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "${LOG_DIR}" "${GUNC_OUT_ROOT}"

NOTE="ACCURACY RUN, not the controlled speed benchmark (WS8.1). Executed on a shared, concurrently loaded 48-core machine (other agents active), so wall-clock is indicative only."

echo "=============================================================="
echo "WS1.5 competitor runs on clean Sets C/D -- $(date -Is)"
echo "=============================================================="
echo "Threads : ${THREADS}"
echo "Tools   : ${TOOLS}"
echo "Sets    : ${SETS}"
echo "Load    : $(cat /proc/loadavg)"
echo ""

has_tool() { [[ ",${TOOLS}," == *",$1,"* ]]; }

write_wallclock() {
    # write_wallclock <file> <tool> <set> <version> <elapsed_s> <started> <finished> <command>
    local f="$1" tool="$2" setname="$3" version="$4" elapsed="$5" started="$6" finished="$7" cmd="$8"
    {
        echo "tool=${tool}"
        echo "version=${version}"
        echo "set=${setname}"
        echo "wall_clock_s=${elapsed}"
        echo "threads=${THREADS}"
        echo "started=${started}"
        echo "finished=${finished}"
        echo "host=$(hostname)"
        echo "n_cpus_total=$(nproc)"
        echo "loadavg_at_start=${LOADAVG_AT_START}"
        echo "command=${cmd}"
        echo "note=${NOTE}"
    } > "${f}"
}

OVERALL_START=$(date +%s)
STATUS_LINES=()

for SET_NAME in ${SETS//,/ }; do
    SET_DIR="${BENCHMARK_DIR}/${SET_NAME}"
    FASTA_DIR="${SET_DIR}/fasta"

    if [[ ! -d "${FASTA_DIR}" ]]; then
        echo "ERROR: ${FASTA_DIR} does not exist" >&2
        exit 1
    fi
    N_GENOMES=$(find "${FASTA_DIR}" -maxdepth 1 -name '*.fasta' | wc -l)
    echo "##############################################################"
    echo "# ${SET_NAME}: ${N_GENOMES} genomes"
    echo "##############################################################"

    # ---------------------------------------------------------------- CheckM2
    if has_tool checkm2; then
        OUT_DIR="${SET_DIR}/checkm2_output"
        QR="${OUT_DIR}/quality_report.tsv"
        WC_FILE="${SET_DIR}/checkm2_wallclock.txt"
        LOG_FILE="${LOG_DIR}/checkm2_${SET_NAME}.log"

        N_DONE=0
        if [[ -f "${QR}" ]]; then N_DONE=$(($(wc -l < "${QR}") - 1)); fi
        N_PKL=$(find "${OUT_DIR}" -maxdepth 1 -name '*.pkl' 2>/dev/null | wc -l)

        if [[ "${N_DONE}" -ge "${N_GENOMES}" && "${N_PKL}" -ge 1 && -f "${WC_FILE}" ]]; then
            echo "[checkm2/${SET_NAME}] SKIP -- ${N_DONE} rows in quality_report.tsv, ${N_PKL} feature-vector PKL(s), wallclock recorded"
            STATUS_LINES+=("checkm2 ${SET_NAME} SKIPPED(complete)")
        else
            echo "[checkm2/${SET_NAME}] running (${N_DONE}/${N_GENOMES} rows, ${N_PKL} PKL present) ..."
            LOADAVG_AT_START=$(cut -d' ' -f1-3 /proc/loadavg | tr ' ' '/')
            STARTED=$(date -Is); T0=$(date +%s)
            CMD="conda run -n ${CHECKM2_ENV} env CHECKM2DB=${CHECKM2_DB} checkm2 predict --threads ${THREADS} -x .fasta --input ${FASTA_DIR} --output-directory ${OUT_DIR} --force --dbg_vectors"
            conda run -n "${CHECKM2_ENV}" env CHECKM2DB="${CHECKM2_DB}" \
                checkm2 predict \
                --threads "${THREADS}" \
                -x .fasta \
                --input "${FASTA_DIR}" \
                --output-directory "${OUT_DIR}" \
                --force \
                --dbg_vectors \
                > "${LOG_FILE}" 2>&1
            RC=$?
            T1=$(date +%s); ELAPSED=$((T1 - T0)); FINISHED=$(date -Is)

            if [[ ${RC} -ne 0 ]]; then
                echo "[checkm2/${SET_NAME}] FAILED rc=${RC}; tail of ${LOG_FILE}:" >&2
                tail -40 "${LOG_FILE}" >&2
                STATUS_LINES+=("checkm2 ${SET_NAME} FAILED(rc=${RC})")
            else
                N_DONE=$(($(wc -l < "${QR}") - 1))
                N_PKL=$(find "${OUT_DIR}" -maxdepth 1 -name '*.pkl' | wc -l)
                write_wallclock "${WC_FILE}" checkm2 "${SET_NAME}" "CheckM2 1.0.1" "${ELAPSED}" "${STARTED}" "${FINISHED}" "${CMD}"
                echo "[checkm2/${SET_NAME}] done in ${ELAPSED}s -- ${N_DONE} rows, ${N_PKL} feature-vector PKL(s)"
                STATUS_LINES+=("checkm2 ${SET_NAME} OK ${ELAPSED}s rows=${N_DONE} pkl=${N_PKL}")
            fi
        fi
    fi

    # ---------------------------------------------------------------- CoCoPyE
    if has_tool cocopye; then
        RAW="${SET_DIR}/cocopye_raw_output.csv"
        WC_FILE="${SET_DIR}/cocopye_wallclock.txt"
        LOG_FILE="${LOG_DIR}/cocopye_${SET_NAME}.log"

        N_DONE=0
        if [[ -f "${RAW}" ]]; then N_DONE=$(($(wc -l < "${RAW}") - 1)); fi

        if [[ "${N_DONE}" -ge "${N_GENOMES}" && -f "${WC_FILE}" ]]; then
            echo "[cocopye/${SET_NAME}] SKIP -- ${N_DONE} rows in cocopye_raw_output.csv, wallclock recorded"
            STATUS_LINES+=("cocopye ${SET_NAME} SKIPPED(complete)")
        else
            echo "[cocopye/${SET_NAME}] running (${N_DONE}/${N_GENOMES} rows present) ..."
            LOADAVG_AT_START=$(cut -d' ' -f1-3 /proc/loadavg | tr ' ' '/')
            STARTED=$(date -Is); T0=$(date +%s)
            CMD="conda run -n ${COCOPYE_ENV} cocopye run -i ${FASTA_DIR} -o ${RAW} -t ${THREADS} -v full"
            conda run -n "${COCOPYE_ENV}" \
                cocopye run \
                -i "${FASTA_DIR}" \
                -o "${RAW}" \
                -t "${THREADS}" \
                -v full \
                > "${LOG_FILE}" 2>&1
            RC=$?
            T1=$(date +%s); ELAPSED=$((T1 - T0)); FINISHED=$(date -Is)

            if [[ ${RC} -ne 0 || ! -f "${RAW}" ]]; then
                echo "[cocopye/${SET_NAME}] FAILED rc=${RC}; tail of ${LOG_FILE}:" >&2
                tail -40 "${LOG_FILE}" >&2
                STATUS_LINES+=("cocopye ${SET_NAME} FAILED(rc=${RC})")
            else
                N_DONE=$(($(wc -l < "${RAW}") - 1))
                write_wallclock "${WC_FILE}" cocopye "${SET_NAME}" "CoCoPyE 0.5.0" "${ELAPSED}" "${STARTED}" "${FINISHED}" "${CMD}"
                echo "[cocopye/${SET_NAME}] done in ${ELAPSED}s -- ${N_DONE} rows"
                STATUS_LINES+=("cocopye ${SET_NAME} OK ${ELAPSED}s rows=${N_DONE}")
            fi
        fi
    fi

    # ------------------------------------------------------------------- GUNC
    if has_tool gunc; then
        G_OUT="${GUNC_OUT_ROOT}/${SET_NAME}"
        NORM="${G_OUT}/gunc_normalized.tsv"
        WC_FILE="${SET_DIR}/gunc_wallclock.txt"
        LOG_FILE="${LOG_DIR}/gunc_${SET_NAME}.log"

        N_DONE=0
        if [[ -f "${NORM}" ]]; then N_DONE=$(($(wc -l < "${NORM}") - 1)); fi

        if [[ "${N_DONE}" -ge "${N_GENOMES}" && -f "${WC_FILE}" ]]; then
            echo "[gunc/${SET_NAME}] SKIP -- ${N_DONE} rows in gunc_normalized.tsv, wallclock recorded"
            STATUS_LINES+=("gunc ${SET_NAME} SKIPPED(complete)")
        elif [[ ! -d "${GUNC_DB_DIR}" ]] || [[ -z "$(find "${GUNC_DB_DIR}" -maxdepth 1 -name '*.dmnd' 2>/dev/null)" ]]; then
            echo "[gunc/${SET_NAME}] SKIP -- no GUNC database in ${GUNC_DB_DIR}"
            STATUS_LINES+=("gunc ${SET_NAME} SKIPPED(no database)")
        elif [[ ! -f "${PROJECT_DIR}/scripts/076_run_gunc.py" ]]; then
            echo "[gunc/${SET_NAME}] SKIP -- scripts/076_run_gunc.py not present"
            STATUS_LINES+=("gunc ${SET_NAME} SKIPPED(no runner)")
        else
            echo "[gunc/${SET_NAME}] running ..."
            LOADAVG_AT_START=$(cut -d' ' -f1-3 /proc/loadavg | tr ' ' '/')
            STARTED=$(date -Is); T0=$(date +%s)
            CMD="python ${PROJECT_DIR}/scripts/076_run_gunc.py --input-dir ${FASTA_DIR} --output-dir ${G_OUT} --extension .fasta --threads ${THREADS}"
            conda run -n "${COCOPYE_ENV}" python "${PROJECT_DIR}/scripts/076_run_gunc.py" \
                --input-dir "${FASTA_DIR}" \
                --output-dir "${G_OUT}" \
                --extension .fasta \
                --threads "${THREADS}" \
                > "${LOG_FILE}" 2>&1
            RC=$?
            T1=$(date +%s); ELAPSED=$((T1 - T0)); FINISHED=$(date -Is)

            if [[ ${RC} -ne 0 || ! -f "${NORM}" ]]; then
                echo "[gunc/${SET_NAME}] FAILED rc=${RC}; tail of ${LOG_FILE}:" >&2
                tail -40 "${LOG_FILE}" >&2
                STATUS_LINES+=("gunc ${SET_NAME} FAILED(rc=${RC})")
            else
                N_DONE=$(($(wc -l < "${NORM}") - 1))
                GV=$(grep -m1 '"gunc_version"' "${G_OUT}/gunc_run_summary.json" 2>/dev/null | sed 's/.*: *"//; s/".*//')
                write_wallclock "${WC_FILE}" gunc "${SET_NAME}" "${GV:-GUNC}" "${ELAPSED}" "${STARTED}" "${FINISHED}" "${CMD}"
                echo "[gunc/${SET_NAME}] done in ${ELAPSED}s -- ${N_DONE} rows"
                STATUS_LINES+=("gunc ${SET_NAME} OK ${ELAPSED}s rows=${N_DONE}")
            fi
        fi
    fi
    echo ""
done

OVERALL_END=$(date +%s)
echo "=============================================================="
echo "Finished at $(date -Is); total $((OVERALL_END - OVERALL_START))s"
for line in "${STATUS_LINES[@]}"; do echo "  ${line}"; done
echo "=============================================================="
