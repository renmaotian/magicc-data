#!/usr/bin/env bash
# WS1.6 orchestration: wait for synthesis -> train -> evaluate, with no idle GPU time.
# Every stage is resumable, so this script can be killed and restarted at any point.
#
#   bash scripts/126_run_holdout_pipeline.sh
#
set -u
cd /path/to/magicc
PY=/path/to/conda/envs/magicc2/bin/python
LOG=logs/revision/ws1.6_pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ---------------------------------------------------------------- 1. synthesis
say "waiting for synthesis (121) to finish all 120 batches + normalization"
while true; do
  NORM=$($PY - <<'EOF' 2>/dev/null || echo no
import json
d = json.load(open('data/holdout/build_checkpoint.json'))
print('yes' if d.get('phases', {}).get('normalized') == 'complete' else 'no')
EOF
)
  [ "$NORM" = "yes" ] && break
  # restart synthesis if it died (resumes from checkpoint)
  if ! pgrep -f "python scripts/121_build" > /dev/null; then
    say "synthesis not running - (re)starting; it resumes from its checkpoint"
    setsid $PY scripts/121_build_holdout_training_data.py --workers 24 \
        >> logs/revision/full_121.out 2>&1 < /dev/null &
    disown
    sleep 30
  fi
  sleep 120
done
say "synthesis complete and normalized"

# ------------------------------------------------------------- 2. eval sets
if [ ! -f data/holdout/eval_sets/manifest.json ] || \
   [ "$($PY -c "import json;print(len(json.load(open('data/holdout/eval_sets/manifest.json'))['groups']))" 2>/dev/null || echo 0)" -lt 7 ]; then
  say "generating evaluation sets (123)"
  $PY scripts/123_generate_holdout_eval_sets.py --workers 16 \
      >> logs/revision/full_123.out 2>&1
fi
say "evaluation sets ready"

# ---------------------------------------------------------------- 3. training
if [ ! -f models/magicc_holdout_phylum.onnx ]; then
  say "starting/resuming holdout training (122) - expect ~5 h"
  RESUME=""
  ls models/holdout/checkpoint_epoch_*.pt > /dev/null 2>&1 && RESUME="--resume-auto"
  $PY scripts/122_train_holdout_phylum.py --seed 42 $RESUME \
      >> logs/revision/ws1.6_train_holdout.log 2>&1
fi
say "training + ONNX export complete"

# -------------------------------------------------------------- 4. evaluation
say "evaluating holdout vs production V5 (124)"
$PY scripts/124_evaluate_holdout.py >> logs/revision/ws1.6_evaluate.log 2>&1
say "evaluation complete"

# ------------------------------------------------- 5. refresh WS1.7 numbers
say "refreshing k-mer reselection control (125) with the full churn analysis"
$PY scripts/125_kmer_reselection_control.py --workers 24 \
    > logs/revision/ws1.7_kmer_reselection.log 2>&1
say "PIPELINE COMPLETE"
