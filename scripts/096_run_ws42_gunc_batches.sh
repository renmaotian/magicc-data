#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# WS4.2 -- collapse-safe, resumable GUNC driver.
#
# Runs a queue of (benchmark set x GUNC database) jobs in priority order.
# Each job is split into fixed-size batches; every batch is an independent
# `scripts/76_run_gunc.py` invocation writing its own normalized TSV, so an
# interruption costs at most one batch. Re-running the script skips every batch
# that already produced a complete normalized TSV, then concatenates the
# batches into runs/<set>/<db>/gunc_normalized.tsv.
#
# Gene-call reuse: if a previous run of the SAME set (any database) left
# per-genome prodigal output behind, the batch is fed those .faa files with
# --gene-calls, which skips gene calling entirely. Prodigal output is
# deterministic, so this changes nothing except wall-clock.
#
#   bash scripts/139_run_ws42_gunc_batches.sh
#   BATCH=250 THREADS=8 bash scripts/139_run_ws42_gunc_batches.sh
#   JOBS="set_E:progenomes_2.1" bash scripts/139_run_ws42_gunc_batches.sh
# ---------------------------------------------------------------------------
set -uo pipefail

PROJ=/path/to/magicc
cd "$PROJ" || exit 1

THREADS=${THREADS:-8}
BATCH=${BATCH:-250}
TMPROOT=${TMPROOT:-/path/to/gunc_tmp}
RUNS=$PROJ/results/revision/gunc/runs
LOG=$PROJ/logs/revision/ws4.2_gunc_driver.log

DB_progenomes_2_1=$PROJ/tools/gunc_db/gunc_db_progenomes2.1.dmnd
DB_gtdb_95=$PROJ/tools/gunc_db_gtdb95/gunc_db_gtdb95.dmnd

# Priority order (protocol WS4.2): clean C/D on BOTH databases first -- exact
# ground truth AND the lineages where the database-coverage question is
# decisive -- then set_E, then the legacy sets.
JOBS=${JOBS:-"set_C_clean:gtdb_95 set_D_clean:gtdb_95 set_E:progenomes_2.1 set_E:gtdb_95 set_A_v2:progenomes_2.1 set_B_v2:progenomes_2.1"}

mkdir -p "$TMPROOT" "$(dirname "$LOG")"

say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

db_path() {
  case "$1" in
    progenomes_2.1) echo "$DB_progenomes_2_1" ;;
    gtdb_95)        echo "$DB_gtdb_95" ;;
    *) echo "" ;;
  esac
}

# Locate an existing per-genome prodigal gene-call directory for this set.
find_gene_calls() {
  local set_name=$1
  for cand in \
      "$PROJ/results/revision/benchmark/gunc/$set_name/gunc_output/gene_calls" \
      "$RUNS/$set_name"/*/gene_calls_cache ; do
    if [ -d "$cand" ] && [ "$(find "$cand" -maxdepth 1 -name '*.genecalls.faa' | head -1)" != "" ]; then
      echo "$cand"; return 0
    fi
  done
  echo ""
}

for job in $JOBS; do
  set_name=${job%%:*}
  db=${job##*:}
  dbf=$(db_path "$db")
  outdir=$RUNS/$set_name/$db
  fasta_dir=$PROJ/data/benchmarks/$set_name/fasta

  if [ -z "$dbf" ] || [ ! -f "$dbf" ]; then say "SKIP $job -- database file missing ($dbf)"; continue; fi
  if [ ! -d "$fasta_dir" ]; then say "SKIP $job -- $fasta_dir missing"; continue; fi

  n_expected=$(find "$fasta_dir" -maxdepth 1 -name '*.fasta' | wc -l)
  if [ -f "$outdir/gunc_normalized.tsv" ]; then
    have=$(( $(wc -l < "$outdir/gunc_normalized.tsv") - 1 ))
    if [ "$have" -eq "$n_expected" ]; then say "SKIP $job -- already complete ($have genomes)"; continue; fi
    say "REDO $job -- existing normalized TSV has $have/$n_expected rows"
  fi

  mkdir -p "$outdir/batches"
  gc_dir=$(find_gene_calls "$set_name")

  # ---- build the per-batch input lists -----------------------------------
  if [ -n "$gc_dir" ]; then
    say "$job: reusing pre-computed gene calls from $gc_dir"
    find "$gc_dir" -maxdepth 1 -name '*.genecalls.faa' | sort > "$outdir/all_inputs.txt"
    GC_FLAG="--gene-calls"
  else
    say "$job: no cached gene calls; prodigal will run"
    find "$fasta_dir" -maxdepth 1 -name '*.fasta' | sort > "$outdir/all_inputs.txt"
    GC_FLAG=""
  fi
  n_in=$(wc -l < "$outdir/all_inputs.txt")
  if [ "$n_in" -ne "$n_expected" ]; then
    say "WARN $job: $n_in input(s) but $n_expected FASTA(s) in the set"
  fi

  # Batches are BALANCED BY REPLICATE, not arbitrary: the clean sets are
  # 100 reference genomes x 10 simulations, so splitting on `replicate` makes
  # every batch a stratified sample containing all 100 references. Any prefix
  # of completed batches is therefore a valid, cluster-balanced subsample --
  # which matters because these runs are long and may be interrupted.
  if ! ls "$outdir"/batches/list_* >/dev/null 2>&1; then
    python - "$outdir/all_inputs.txt" "$PROJ/data/benchmarks/$set_name/metadata.tsv" \
             "$outdir/batches" "$BATCH" <<'PYEOF'
import os, sys
inputs, meta_p, bdir, batch = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
paths = [l.strip() for l in open(inputs) if l.strip()]


def gname(p):
    n = os.path.basename(p)
    for suf in ('.genecalls.faa', '.fasta', '.fna', '.fa', '.faa'):
        if n.endswith(suf):
            return n[:-len(suf)]
    return os.path.splitext(n)[0]


rep = {}
if os.path.isfile(meta_p):
    with open(meta_p) as f:
        hdr = f.readline().rstrip('\n').split('\t')
        if 'replicate' in hdr and 'genome_id' in hdr:
            gi, ri = hdr.index('genome_id'), hdr.index('replicate')
            for line in f:
                c = line.rstrip('\n').split('\t')
                if len(c) > max(gi, ri):
                    try:
                        rep[c[gi]] = int(c[ri])
                    except ValueError:
                        pass

n_batches = max(1, -(-len(paths) // batch))
if rep and all(gname(p) in rep for p in paths):
    reps = sorted({rep[gname(p)] for p in paths})
    per = max(1, -(-len(reps) // n_batches))
    groups = [set(reps[i:i + per]) for i in range(0, len(reps), per)]
    buckets = [[p for p in paths if rep[gname(p)] in g] for g in groups]
    mode = f'balanced by replicate ({len(reps)} replicates -> {len(buckets)} batches)'
else:
    # No replicate column (set_E / A_v2 / B_v2): genome_id order tracks
    # sample category, so contiguous chunks would be badly unrepresentative.
    # Round-robin over the numerically ordered list instead, which gives every
    # batch a proportional mix of categories.
    def gidx(p):
        s = gname(p).rsplit('_', 1)[-1]
        return int(s) if s.isdigit() else 0
    ordered = sorted(paths, key=gidx)
    buckets = [ordered[i::n_batches] for i in range(n_batches)]
    mode = f'round-robin over genome index ({n_batches} batches)'

os.makedirs(bdir, exist_ok=True)
for i, b in enumerate(buckets):
    if b:
        open(os.path.join(bdir, f'list_{i:03d}'), 'w').write('\n'.join(b) + '\n')
print(f'  split: {mode}; sizes={[len(b) for b in buckets]}')
PYEOF
  fi

  # ---- run every batch ----------------------------------------------------
  job_rc=0
  for lst in "$outdir"/batches/list_*; do
    b=$(basename "$lst"); b=${b#list_}
    bdir=$outdir/batches/batch_$b
    want=$(wc -l < "$lst")
    if [ -f "$bdir/gunc_normalized.tsv" ]; then
      got=$(( $(wc -l < "$bdir/gunc_normalized.tsv") - 1 ))
      if [ "$got" -eq "$want" ]; then say "  batch $b: done ($got)"; continue; fi
      say "  batch $b: incomplete ($got/$want) -- rerunning"
      rm -rf "$bdir"
    fi
    say "  batch $b: START ($want genomes, $db, ${THREADS}t)"
    t0=$(date +%s)
    python "$PROJ/scripts/76_run_gunc.py" $GC_FLAG \
        --input-list "$lst" --output-dir "$bdir" \
        --db "$dbf" --threads "$THREADS" \
        --temp-dir "$TMPROOT" >> "$LOG" 2>&1
    rc=$?
    t1=$(date +%s)
    if [ $rc -ne 0 ]; then say "  batch $b: FAILED rc=$rc after $((t1-t0))s"; job_rc=1; continue; fi
    say "  batch $b: END $((t1-t0))s ($(( (t1-t0) * 1000 / want ))ms/genome)"
    # Cache the gene calls from the first run of this set so the second
    # database does not have to re-run prodigal.
    if [ -z "$GC_FLAG" ] && [ -d "$bdir/gunc_output/gene_calls" ]; then
      mkdir -p "$outdir/gene_calls_cache"
      find "$bdir/gunc_output/gene_calls" -maxdepth 1 -name '*.genecalls.faa' \
        -exec cp -n {} "$outdir/gene_calls_cache/" \; 2>/dev/null
    fi
    # DIAMOND raw hits are large and are not needed downstream.
    rm -rf "$bdir/gunc_output/diamond_output"
  done

  # ---- concatenate --------------------------------------------------------
  first=1; tmp=$outdir/.gunc_normalized.tsv.part; : > "$tmp"
  for bdir in "$outdir"/batches/batch_*; do
    [ -f "$bdir/gunc_normalized.tsv" ] || continue
    if [ $first -eq 1 ]; then cat "$bdir/gunc_normalized.tsv" >> "$tmp"; first=0
    else tail -n +2 "$bdir/gunc_normalized.tsv" >> "$tmp"; fi
  done
  rows=$(( $(wc -l < "$tmp") - 1 ))
  if [ "$rows" -eq "$n_expected" ]; then
    mv "$tmp" "$outdir/gunc_normalized.tsv"
    say "$job: COMPLETE -- $rows genomes -> $outdir/gunc_normalized.tsv"
  else
    mv "$tmp" "$outdir/gunc_normalized.partial.tsv"
    say "$job: PARTIAL -- $rows/$n_expected genomes (rc=$job_rc); rerun to resume"
  fi
done

say "DRIVER_DONE"
