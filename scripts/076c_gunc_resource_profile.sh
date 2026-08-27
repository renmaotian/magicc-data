#!/bin/bash
# WS4.1 / WS8.1(prep) -- GUNC runtime, peak memory and thread scaling.
#
# Runs the SAME 6-genome control set at 1, 2 and 4 threads (the 8-thread point is
# measured separately), then a 50-genome batch at 8 threads to quantify how the
# large fixed per-invocation DIAMOND database scan amortizes with batch size.
#
# All runs use /usr/bin/time -v so peak RSS covers the whole process tree
# (python wrapper + prodigal workers + diamond).
#
# NOTE: capped at 8 threads because another agent shares this machine. The
# definitive matched-hardware benchmark (1/8/16/32 threads on an idle host) is
# WS8.1.
set -u
PROJ=/path/to/magicc
OUT=$PROJ/results/revision/gunc/resource_profile
SMALL=$PROJ/results/revision/gunc/controls/negative_fasta
BATCH=$OUT/batch50_fasta
mkdir -p "$OUT"

source /path/to/anaconda3/etc/profile.d/conda.sh
conda activate magicc2

run_one () {   # run_one <label> <input_dir> <threads>
  local LABEL=$1 IN=$2 T=$3
  local RUN=$OUT/$LABEL
  rm -rf "$RUN"; mkdir -p "$RUN"
  echo "--- $LABEL : threads=$T  input=$IN ($(ls "$IN" | wc -l) genomes) ---"
  date
  /usr/bin/time -v -o "$OUT/time_${LABEL}.txt" \
    python "$PROJ/scripts/076_run_gunc.py" \
      --input-dir "$IN" --output-dir "$RUN" \
      --extension .fna --threads "$T" \
      > "$OUT/stdout_${LABEL}.log" 2>&1
  echo "  exit=$?"
  grep -E 'Elapsed \(wall clock\)|Maximum resident set size|Percent of CPU|User time|System time' \
    "$OUT/time_${LABEL}.txt" | sed 's/^/  /'
  # phase breakdown from the GUNC log
  awk '/START Prodigal|END   Prodigal|START Diamond|END   Diamond|END   Runtime/' \
    "$RUN/gunc_run.log" 2>/dev/null | sed 's/^/  /'
}

# ---- batch amortization: 50 genomes at 8 threads (run first: most useful) ----
mkdir -p "$BATCH"
i=0
for f in "$PROJ"/data/ncbi/pure_culture/*.fna; do
  ln -sf "$f" "$BATCH/$(basename "$f")"
  i=$((i+1)); [ $i -ge 50 ] && break
done
run_one "batch50_t8" "$BATCH" 8

# ---- thread scaling on the fixed 6-genome control set -------------------
for T in 4 2 1; do
  run_one "small_t${T}" "$SMALL" "$T"
done

echo
echo "=== ALL RUNS COMPLETE ==="
date
