#!/usr/bin/env bash
# WS11.G orchestration: evaluation sets -> synthesis -> training -> evaluation ->
# genus/family/phylum ladder -> k-mer control -> report, with no idle GPU time.
#
# Every stage is resumable and idempotent, so this script can be killed and
# restarted at any point and will pick up where it left off. It is designed to
# run detached (setsid/nohup) and finish with no agent attached.
#
#   setsid nohup bash scripts/221_run_holdout_genus_pipeline.sh \
#       >> logs/revision/ws11g_pipeline.out 2>&1 < /dev/null &
#
set -u
cd /path/to/magicc
export MAGICC_HOLDOUT_LEVEL=genus
export PYTHONHASHSEED=0
PY=/path/to/conda/bin/python
LOG=logs/revision/ws11g_pipeline.log
WORKERS=30            # cap: other heavy jobs share this 48-core box
RES=results/revision/holdout_genus
mkdir -p logs/revision "$RES"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "================ WS11.G leave-GENUS-out pipeline ================"
say "panel: genera held out from FAMILIES and phyla that REMAIN in training"

# ------------------------------------------------------------------ 0. EDA
if [ ! -f data/holdout_genus/holdout_panel.json ]; then
  say "running panel EDA (220)"
  $PY scripts/220_holdout_genus_panel_eda.py >> logs/revision/ws11g_eda.log 2>&1 \
    || { say "EDA FAILED"; exit 1; }
fi
say "panel definition ready"

# ------------------------------------------------ 1. synthesis starts FIRST
# It is the long pole, so it goes to the background immediately and the cheap
# CPU stages run alongside it.
SYNTH_PID=logs/revision/ws11g_synth.pid
# NOTE: `pgrep -f` MUST NOT be used to detect the synthesis job. This script own
# text contains the script name, so any shell launched with the script inlined
# matches the pattern, the guard concludes "already running", and synthesis is
# silently never started. That happened on 2026-08-27. Detection uses a PID file
# with a fallback that inspects only PYTHON processes.
synth_running() {
  if [ -s "$SYNTH_PID" ] && kill -0 "$(cat "$SYNTH_PID")" 2>/dev/null; then return 0; fi
  P=$(ps -eo pid,args | awk '$2 ~ /python$/ && /121_build_ho/ && !/awk/ {print $1}' | head -1)
  [ -n "$P" ] && { echo "$P" > "$SYNTH_PID"; return 0; }
  return 1
}
start_synth() {
  MAGICC_HOLDOUT_LEVEL=genus PYTHONHASHSEED=0 \
    setsid $PY scripts/121_build_holdout_training_data.py --workers $WORKERS \
      >> logs/revision/ws11g_full_121.out 2>&1 < /dev/null &
  disown
  sleep 10
  ps -eo pid,args | awk '$2 ~ /python$/ && /121_build_ho/ && !/awk/ {print $1}' | head -1 > "$SYNTH_PID"
}
norm_done() {
  $PY - <<'EOF' 2>/dev/null || echo no
import json, os
p = 'data/holdout_genus/build_checkpoint.json'
print('yes' if os.path.exists(p) and json.load(open(p)).get('phases', {})
      .get('normalized') == 'complete' else 'no')
EOF
}
if ! synth_running; then
  if [ "$(norm_done)" != "yes" ]; then
    say "launching synthesis (121) in the background with $WORKERS workers"
    start_synth
    sleep 20
  fi
fi

# ------------------------------------- 1a. evaluation sets (CPU, concurrent)
# Cheap, and the earliest hard failure point; also enables the V5-vs-V5 dry-run
# validation while synthesis is still running.
NG=$($PY -c "import json;print(len(json.load(open('data/holdout_genus/eval_sets/manifest.json'))['groups']))" 2>/dev/null || echo 0)
if [ "$NG" -lt 7 ]; then
  say "generating evaluation sets (123), concurrent with synthesis"
  $PY scripts/123_generate_holdout_eval_sets.py --workers 8 \
      >> logs/revision/ws11g_full_123.out 2>&1
fi
say "evaluation sets ready ($($PY -c "import json;print(len(json.load(open('data/holdout_genus/eval_sets/manifest.json'))['groups']))" 2>/dev/null || echo 0) groups)"

# ------------------------------------------ 1b. DRY-RUN DESIGN VALIDATION
# Substituting production V5 for BOTH models must give exactly zero for every
# difference. WS1.6 and WS1.9 both did this and both passed.
if [ ! -f $RES/dryrun_v5_vs_v5/lineage_novelty_effect_did.tsv ]; then
  say "dry-run validation: production V5 substituted for both models"
  $PY scripts/124_evaluate_holdout.py \
      --holdout-onnx models/magicc_v5.onnx \
      --holdout-norm data/features/normalization_params.json \
      --out-dir $RES/dryrun_v5_vs_v5 \
      >> logs/revision/ws11g_dryrun.log 2>&1
fi
$PY - <<'EOF' 2>&1 | tee -a "$LOG"
import pandas as pd, pathlib
p = pathlib.Path('results/revision/holdout_genus/dryrun_v5_vs_v5/'
                 'lineage_novelty_effect_did.tsv')
if p.exists():
    d = pd.read_csv(p, sep='\t')
    cols = [c for c in d.columns if c.endswith(('_did', '_delta'))]
    mx = float(d[cols].abs().to_numpy().max())
    print(f'DRY-RUN VALIDATION: max |difference| over {len(cols)} columns {cols} '
          f'= {mx:.3e} -> {"PASS (exactly zero)" if mx == 0.0 else "FAIL"}')
EOF

# ------------------------------------------------- 2. wait for synthesis (CPU)
say "waiting for synthesis (121): 120 batches + normalization"
while true; do
  [ "$(norm_done)" = "yes" ] && break
  # restart synthesis if it died (it resumes from its per-batch checkpoint)
  if ! synth_running; then
    say "synthesis not running - (re)starting; resumes from checkpoint"
    start_synth
    sleep 30
  fi
  sleep 120
done
say "synthesis complete and normalized"

# ---------------------------------------------------------- 3. training (GPU)
if [ ! -f models/magicc_holdout_genus.onnx ]; then
  say "starting/resuming holdout training (122) - seed 42"
  for ATTEMPT in 1 2 3; do
    RESUME=""
    ls models/holdout_genus/checkpoint_epoch_*.pt > /dev/null 2>&1 && RESUME="--resume-auto"
    $PY scripts/122_train_holdout_phylum.py --seed 42 $RESUME \
        >> logs/revision/ws11g_train_holdout.log 2>&1
    [ -f models/magicc_holdout_genus.onnx ] && break
    say "training attempt $ATTEMPT did not produce the ONNX; retrying from checkpoint"
    sleep 60
  done
fi
[ -f models/magicc_holdout_genus.onnx ] || { say "TRAINING FAILED - no ONNX"; exit 1; }
say "training + ONNX export complete"

# -------------------------------------------------------------- 4. evaluation
say "evaluating holdout vs production V5 (124) on identical samples"
$PY scripts/124_evaluate_holdout.py >> logs/revision/ws11g_evaluate.log 2>&1 \
  || { say "EVALUATION FAILED"; exit 1; }
say "evaluation complete"

# ------------------------- 5. genus vs family vs phylum ladder (four models)
if [ -f scripts/223_genus_family_phylum_ladder.py ]; then
  say "genus-vs-family-vs-phylum ladder (223)"
  $PY scripts/223_genus_family_phylum_ladder.py \
      >> logs/revision/ws11g_ladder.log 2>&1 \
    || say "ladder step failed (non-fatal; evaluation outputs are already written)"
else
  say "223 not present yet - skipping ladder"
fi

# -------------------------------------------- 6. k-mer reselection control
if [ ! -f $RES/kmer_reselection_summary.json ]; then
  say "k-mer reselection control (125)"
  $PY scripts/125_kmer_reselection_control.py --workers $WORKERS \
      >> logs/revision/ws11g_kmer_reselection.log 2>&1 \
    || say "k-mer control failed (non-fatal)"
fi

# ------------------------------------------------------------------ 7. report
if [ -f scripts/225_ws11g_report.py ]; then
  say "assembling WS11_G_REPORT.md (225)"
  $PY scripts/225_ws11g_report.py >> logs/revision/ws11g_report.log 2>&1 \
    || say "report step failed (non-fatal)"
fi

say "PIPELINE COMPLETE - results in $RES/"
