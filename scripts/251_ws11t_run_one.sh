#!/bin/bash
# WS11.T — execute ONE timed cell.  Derived from scripts/161_ws8_run_one.sh; the
# competitor/MAGICC/DeepCheck dispatch blocks are byte-identical to 161 so the
# new repeats are measured by exactly the harness that produced the archived
# rows.  Differences from 161, all deliberate:
#
#   * results are written under results/revision/speed_v3/ (161's tree is cited
#     by the current manuscript and must not be touched);
#   * the INPUT SET is read from results/revision/speed/inputs/ — the SAME
#     files, the same manifest, so the denominators are identical;
#   * host state is recorded before AND after every cell: all three load
#     averages, nproc, MemAvailable and `vmstat`;
#   * five extra MAGICC arms exist for the V3-vs-V5 attribution (T3).
#
#   bash scripts/191_ws11t_run_one.sh <tool> <threads> <repeat> <inputset>
#
#     tool : magicc | magicc_dir | magicc_v5code | magicc_v3code
#            | magicc_v3code_nostats | checkm2 | cocopye | deepcheck
#     inputset : set_E_100 | set_E_full | set_E_full_dir
#
# WALL-CLOCK DEFINITION (identical for every tool, identical to WS8):
#   process start -> results file written & process exited, INCLUDING
#   interpreter start, imports, JIT warm-up, model/database load, output write.
#   Instrument: /usr/bin/time -v on the tool's own top-level command.
#
# Idempotent/resumable: an existing complete .json for the cell is not re-run.
set -uo pipefail

PROJECT="/path/to/magicc"
ENVS="/path/to/anaconda3/envs"
CHECKM2_DB="${PROJECT}/tools/checkm2_db/CheckM2_database/uniref100.KO.1.dmnd"
SPEED_SRC="${PROJECT}/results/revision/speed"          # inputs only, READ-ONLY
SPEED="${PROJECT}/results/revision/speed_v3"           # all new outputs
RUNS="${SPEED}/runs"
SCRATCH="${SPEED}/scratch"
V3CODE="${SPEED}/v3_code"
V3CODE_NOSTATS="${SPEED}/v3_code_nostats"
LAUNCHER="${PROJECT}/scripts/190_ws11t_run_magicc_codebase.py"

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

# ---- resolve the input set.  set_E_full_dir is the SAME 1,000 genomes as
# ---- set_E_full, supplied as a directory (verified identical file-for-file);
# ---- it exists because the archived V3 CLI had no --input-list option.
case "${INPUTSET}" in
  set_E_full_dir) INLIST="${SPEED_SRC}/inputs/set_E_full.txt" ;;
  *)              INLIST="${SPEED_SRC}/inputs/${INPUTSET}.txt" ;;
esac
INDIR="${SPEED_SRC}/inputs/${INPUTSET}"
if [ "${INPUTSET}" = "set_E_full" ] || [ "${INPUTSET}" = "set_E_full_dir" ]; then
    INDIR="${PROJECT}/data/benchmarks/set_E/fasta"
fi
NGEN=$(wc -l < "${INLIST}")

WORK="${SCRATCH}/${CELL}"
rm -rf "${WORK}"; mkdir -p "${WORK}"

CODE_ROOT=""
MODEL="${PROJECT}/models/magicc_v5.onnx"

# ------------------------------------------------------------- host state
snap_mem() { awk '/^MemAvailable:/{print $2}' /proc/meminfo; }
snap_vm()  { vmstat 1 2 | tail -n 1 | sed 's/^[[:space:]]*//;s/[[:space:]]\+/ /g'; }

LOAD_BEFORE=$(cut -d' ' -f1-3 /proc/loadavg)
MEM_BEFORE=$(snap_mem)
VM_BEFORE=$(snap_vm)
NPROC=$(nproc)
NPROC_BUSY_BEFORE=$(ps -eo pcpu= | awk '{s+=$1} END {printf "%.1f", s}')
T_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# ---------------------------------------------------------------- dispatch
case "${TOOL}" in

  magicc)
    # PRODUCTION console script, --input-list, V5.  Byte-identical to 161.
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

  magicc_dir)
    # T3 ladder rung 2: production console script, V5, but DIRECTORY input with
    # --extension .fasta, i.e. exactly the input mode of the archived V3 runs.
    # Isolates input-discovery mode (list vs directory scan).
    OUTF="${WORK}/magicc_predictions.tsv"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /usr/bin/time -v -o "${TIMEF}" \
        "${ENVS}/magicc2/bin/magicc" predict \
            --input "${INDIR}" \
            --extension .fasta \
            --output "${OUTF}" \
            --threads "${THREADS}" \
            --batch-size 64 \
            --model "${PROJECT}/models/magicc_v5.onnx" \
        > "${STDOUTF}" 2>&1
    RC=$?
    NOUT=$(( $(wc -l < "${OUTF}" 2>/dev/null || echo 1) - 1 ))
    ;;

  magicc_v5code|magicc_v3code|magicc_v3code_nostats)
    # T3 ladder rungs 3-5.  All three go through the SAME launcher, so the
    # launcher's own cost is common-mode and cancels in the contrasts.
    case "${TOOL}" in
      magicc_v5code)          CODE_ROOT="${PROJECT}";          MODEL="${PROJECT}/models/magicc_v5.onnx" ;;
      magicc_v3code)          CODE_ROOT="${V3CODE}";           MODEL="${PROJECT}/models/magicc_v3.onnx" ;;
      magicc_v3code_nostats)  CODE_ROOT="${V3CODE_NOSTATS}";   MODEL="${PROJECT}/models/magicc_v3.onnx" ;;
    esac
    OUTF="${WORK}/magicc_predictions.tsv"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /usr/bin/time -v -o "${TIMEF}" \
        "${ENVS}/magicc2/bin/python" "${LAUNCHER}" \
            --code-root "${CODE_ROOT}" -- \
            predict \
            --input "${INDIR}" \
            --extension .fasta \
            --output "${OUTF}" \
            --threads "${THREADS}" \
            --batch-size 64 \
            --model "${MODEL}" \
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
    OUTF="${WORK}/deepcheck_predictions.tsv"
    /usr/bin/time -v -o "${TIMEF}" \
        env OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
        "${ENVS}/magicc2/bin/python" "${PROJECT}/scripts/163_ws8_deepcheck_infer.py" \
            --features "${SPEED_SRC}/deepcheck_features/${INPUTSET}" \
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
MEM_AFTER=$(snap_mem)
VM_AFTER=$(snap_vm)

python3 "${PROJECT}/scripts/192_ws11t_parse_time.py" \
    --time-file "${TIMEF}" \
    --json "${JSON}" \
    --tool "${TOOL}" --threads "${THREADS}" --repeat "${REP}" \
    --input-set "${INPUTSET}" --cache "${CACHE}" \
    --n-genomes "${NGEN}" --n-output "${NOUT}" \
    --rc "${RC}" \
    --t-start "${T_START}" --t-end "${T_END}" \
    --load-before "${LOAD_BEFORE}" --load-after "${LOAD_AFTER}" \
    --busy-before "${NPROC_BUSY_BEFORE}" \
    --nproc "${NPROC}" \
    --mem-avail-before "${MEM_BEFORE}" --mem-avail-after "${MEM_AFTER}" \
    --vmstat-before "${VM_BEFORE}" --vmstat-after "${VM_AFTER}" \
    --code-root "${CODE_ROOT}" --model "${MODEL}" \
    --campaign "WS11.T"

# keep outputs small: drop the big intermediate trees, keep the result table
if [ "${TOOL}" = "checkm2" ]; then
    cp -f "${WORK}/checkm2_out/quality_report.tsv" "${RUNS}/${CELL}.result.tsv" 2>/dev/null
elif [ "${TOOL}" = "cocopye" ]; then
    cp -f "${WORK}/cocopye_output.csv" "${RUNS}/${CELL}.result.csv" 2>/dev/null
else
    cp -f "${WORK}"/*predictions.tsv "${RUNS}/${CELL}.result.tsv" 2>/dev/null
fi
rm -rf "${WORK}"

echo "[done] ${CELL} rc=${RC}"
exit 0
