#!/bin/bash
# WS11.T / v0.3.3 — execute ONE timed MAGICC cell under the RELEASED code path.
#
# Derived from scripts/251_ws11t_run_one.sh; identical wall-clock definition,
# identical input set (read from results/revision/speed/inputs/), identical
# host-state recording.  Outputs go to results/revision/speed_v033/.
#
#   bash scripts/259_ws11t_v033_run_one.sh <tool> <threads> <repeat> <inputset>
#
# TWO ARMS, differing by exactly one thing -- whether --model is given:
#
#   magicc_v033        `magicc predict` with NO --model.  The CLI resolves the
#                      frozen V5 model itself (package data -> project layout ->
#                      ~/.magicc) and, since 0.3.2, VERIFIES ITS SHA256 ON EVERY
#                      RUN before loading it.  THIS IS WHAT A RELEASED USER GETS.
#
#   magicc_v033_xmodel `magicc predict --model <path>`.  Documented in --help as
#                      "used as given and is NOT checksum-verified", so that
#                      alternative models can be evaluated deliberately.  This is
#                      byte-for-byte the invocation the WS11.T campaign used.
#
# Both resolve to the SAME FILE (models/magicc_v5.onnx, 169,658,949 B, SHA256
# b843466...b3096), so the arms differ only by the verification pass and the
# A - B contrast is the cost of that pass and nothing else.
set -uo pipefail

PROJECT="/path/to/magicc"
ENVS="/path/to/anaconda3/envs"
SPEED_SRC="${PROJECT}/results/revision/speed"        # inputs only, READ-ONLY
SPEED="${PROJECT}/results/revision/speed_v033"
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
    echo "[skip] ${CELL} already complete"; exit 0
fi

INLIST="${SPEED_SRC}/inputs/${INPUTSET}.txt"
NGEN=$(wc -l < "${INLIST}")
WORK="${SCRATCH}/${CELL}"; rm -rf "${WORK}"; mkdir -p "${WORK}"
OUTF="${WORK}/magicc_predictions.tsv"

snap_mem() { awk '/^MemAvailable:/{print $2}' /proc/meminfo; }
snap_vm()  { vmstat 1 2 | tail -n 1 | sed 's/^[[:space:]]*//;s/[[:space:]]\+/ /g'; }

LOAD_BEFORE=$(cut -d' ' -f1-3 /proc/loadavg)
MEM_BEFORE=$(snap_mem); VM_BEFORE=$(snap_vm); NPROC=$(nproc)
NPROC_BUSY_BEFORE=$(ps -eo pcpu= | awk '{s+=$1} END {printf "%.1f", s}')
T_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
MODEL_ARG="(default resolution, SHA256 verified every run)"

case "${TOOL}" in
  magicc_v033)
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /usr/bin/time -v -o "${TIMEF}" \
        "${ENVS}/magicc2/bin/magicc" predict \
            --input-list "${INLIST}" --output "${OUTF}" --threads "${THREADS}" \
        > "${STDOUTF}" 2>&1
    RC=$? ;;
  magicc_v033_xmodel)
    MODEL_ARG="${PROJECT}/models/magicc_v5.onnx (explicit --model, NOT verified)"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /usr/bin/time -v -o "${TIMEF}" \
        "${ENVS}/magicc2/bin/magicc" predict \
            --input-list "${INLIST}" --output "${OUTF}" --threads "${THREADS}" \
            --model "${PROJECT}/models/magicc_v5.onnx" \
        > "${STDOUTF}" 2>&1
    RC=$? ;;
  *) echo "unknown tool: ${TOOL}" >&2; exit 2 ;;
esac
NOUT=$(( $(wc -l < "${OUTF}" 2>/dev/null || echo 1) - 1 ))

T_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
LOAD_AFTER=$(cut -d' ' -f1-3 /proc/loadavg)
MEM_AFTER=$(snap_mem); VM_AFTER=$(snap_vm)

python3 "${PROJECT}/scripts/252_ws11t_parse_time.py" \
    --time-file "${TIMEF}" --json "${JSON}" \
    --tool "${TOOL}" --threads "${THREADS}" --repeat "${REP}" \
    --input-set "${INPUTSET}" --cache "${CACHE}" \
    --n-genomes "${NGEN}" --n-output "${NOUT}" --rc "${RC}" \
    --t-start "${T_START}" --t-end "${T_END}" \
    --load-before "${LOAD_BEFORE}" --load-after "${LOAD_AFTER}" \
    --busy-before "${NPROC_BUSY_BEFORE}" --nproc "${NPROC}" \
    --mem-avail-before "${MEM_BEFORE}" --mem-avail-after "${MEM_AFTER}" \
    --vmstat-before "${VM_BEFORE}" --vmstat-after "${VM_AFTER}" \
    --code-root "${PROJECT} (magicc 0.3.3)" --model "${MODEL_ARG}" \
    --campaign "WS11.T-v0.3.3"

cp -f "${OUTF}" "${RUNS}/${CELL}.result.tsv" 2>/dev/null
rm -rf "${WORK}"
echo "[done] ${CELL} rc=${RC}"
exit 0
