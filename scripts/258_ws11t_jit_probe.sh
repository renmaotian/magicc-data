#!/bin/bash
# WS11.T (T3 support) — rule the Numba JIT cache in or out as an explanation of
# the V3-vs-V5 gap, by MEASUREMENT rather than argument.
#
# Both MAGICC code trees compile Numba kernels on first use and cache them in
# the package's __pycache__ (.nbi/.nbc).  Every campaign run is JIT-warm.  This
# probe makes a pristine copy of each tree with NO Numba cache and times one
# end-to-end 1-thread run of the 1,000-genome set, so the one-off compile cost
# is measured directly and can be compared with the 25 s gap.
#
#   bash scripts/198_ws11t_jit_probe.sh
set -uo pipefail
PROJECT="/path/to/magicc"
ENVS="/path/to/anaconda3/envs"
SPEED_SRC="${PROJECT}/results/revision/speed"
SPEED="${PROJECT}/results/revision/speed_v3"
RUNS="${SPEED}/runs"
INDIR="${PROJECT}/data/benchmarks/set_E/fasta"
INLIST="${SPEED_SRC}/inputs/set_E_full.txt"
LAUNCHER="${PROJECT}/scripts/190_ws11t_run_magicc_codebase.py"
mkdir -p "${RUNS}" "${SPEED}/scratch"

run_probe () {
    local NAME="$1" SRCTREE="$2" MODEL="$3"
    local TREE="${SPEED}/${NAME}_tree"
    local CELL="${NAME}__t1__set_E_full_dir__warm__r1"
    local JSON="${RUNS}/${CELL}.json"
    if [ -s "${JSON}" ] && grep -q '"wall_clock_s"' "${JSON}" 2>/dev/null; then
        echo "[skip] ${CELL}"; return 0
    fi
    rm -rf "${TREE}"; mkdir -p "${TREE}"
    cp -r "${SRCTREE}" "${TREE}/magicc"
    # pristine: no bytecode, no Numba cache
    find "${TREE}" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null
    echo "[probe] ${CELL}  tree=${TREE}  numba cache files: $(find "${TREE}" -name '*.nb[ic]' | wc -l)"

    local WORK="${SPEED}/scratch/${CELL}"
    rm -rf "${WORK}"; mkdir -p "${WORK}"
    local LOAD_BEFORE MEM_BEFORE VM_BEFORE LOAD_AFTER MEM_AFTER VM_AFTER T_START T_END RC NOUT
    LOAD_BEFORE=$(cut -d' ' -f1-3 /proc/loadavg)
    MEM_BEFORE=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
    VM_BEFORE=$(vmstat 1 2 | tail -n 1 | sed 's/^[[:space:]]*//;s/[[:space:]]\+/ /g')
    T_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /usr/bin/time -v -o "${RUNS}/${CELL}.time.txt" \
        "${ENVS}/magicc2/bin/python" "${LAUNCHER}" --code-root "${TREE}" -- \
            predict --input "${INDIR}" --extension .fasta \
            --output "${WORK}/magicc_predictions.tsv" --threads 1 \
            --batch-size 64 --model "${MODEL}" \
        > "${RUNS}/${CELL}.stdout.txt" 2>&1
    RC=$?
    T_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    LOAD_AFTER=$(cut -d' ' -f1-3 /proc/loadavg)
    MEM_AFTER=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
    VM_AFTER=$(vmstat 1 2 | tail -n 1 | sed 's/^[[:space:]]*//;s/[[:space:]]\+/ /g')
    NOUT=$(( $(wc -l < "${WORK}/magicc_predictions.tsv" 2>/dev/null || echo 1) - 1 ))
    python3 "${PROJECT}/scripts/192_ws11t_parse_time.py" \
        --time-file "${RUNS}/${CELL}.time.txt" --json "${JSON}" \
        --tool "${NAME}" --threads 1 --repeat 1 \
        --input-set set_E_full_dir --cache warm \
        --n-genomes "$(wc -l < "${INLIST}")" --n-output "${NOUT}" --rc "${RC}" \
        --t-start "${T_START}" --t-end "${T_END}" \
        --load-before "${LOAD_BEFORE}" --load-after "${LOAD_AFTER}" \
        --busy-before "$(ps -eo pcpu= | awk '{s+=$1} END {printf "%.1f", s}')" \
        --nproc "$(nproc)" --mem-avail-before "${MEM_BEFORE}" --mem-avail-after "${MEM_AFTER}" \
        --vmstat-before "${VM_BEFORE}" --vmstat-after "${VM_AFTER}" \
        --code-root "${TREE}" --model "${MODEL}" --campaign "WS11.T"
    cp -f "${WORK}/magicc_predictions.tsv" "${RUNS}/${CELL}.result.tsv" 2>/dev/null
    rm -rf "${WORK}"
    echo "[done] ${CELL} rc=${RC}"
}

run_probe magicc_v3code_coldjit "${SPEED}/v3_code/magicc" "${PROJECT}/models/magicc_v3.onnx"
run_probe magicc_v5code_coldjit "${PROJECT}/magicc"       "${PROJECT}/models/magicc_v5.onnx"
