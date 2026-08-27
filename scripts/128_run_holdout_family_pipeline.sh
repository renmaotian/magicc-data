#!/usr/bin/env bash
# WS1.9 orchestration: evaluation sets -> synthesis -> training -> evaluation ->
# family-vs-phylum comparison, with no idle GPU time.
#
# Every stage is resumable and idempotent, so this script can be killed and
# restarted at any point and will pick up where it left off. It is designed to
# run detached (setsid/nohup) and finish with no agent attached.
#
#   setsid nohup bash scripts/128_run_holdout_family_pipeline.sh \
#       >> logs/revision/ws1.9_pipeline.out 2>&1 < /dev/null &
#
set -u
cd /path/to/magicc
export MAGICC_HOLDOUT_LEVEL=family
export PYTHONHASHSEED=0
PY=/path/to/conda/envs/magicc2/bin/python
LOG=logs/revision/ws1.9_pipeline.log
WORKERS=24            # cap: a GUNC job (~8 threads) and other CPU agents share the box
mkdir -p logs/revision
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "================ WS1.9 leave-FAMILY-out pipeline ================"
say "panel: families held out from phyla that REMAIN in training"

# ------------------------------------------------------------------ 0. EDA
if [ ! -f data/holdout_family/holdout_panel.json ]; then
  say "running panel EDA (127)"
  $PY scripts/127_holdout_family_panel_eda.py >> logs/revision/ws1.9_eda.log 2>&1 \
    || { say "EDA FAILED"; exit 1; }
fi
say "panel definition ready"

# ------------------------------------------------ 1. synthesis starts FIRST
# It is the ~7 h long pole, so it goes to the background immediately and the
# cheap CPU stages run alongside it.
if ! pgrep -f "121_build_holdout_training_data" > /dev/null; then
  NORM0=$($PY - <<'EOF' 2>/dev/null || echo no
import json, os
p = 'data/holdout_family/build_checkpoint.json'
print('yes' if os.path.exists(p) and json.load(open(p)).get('phases', {})
      .get('normalized') == 'complete' else 'no')
EOF
)
  if [ "$NORM0" != "yes" ]; then
    say "launching synthesis (121) in the background with $WORKERS workers"
    MAGICC_HOLDOUT_LEVEL=family PYTHONHASHSEED=0 \
      setsid $PY scripts/121_build_holdout_training_data.py --workers $WORKERS \
        >> logs/revision/ws1.9_full_121.out 2>&1 < /dev/null &
    disown
    sleep 20
  fi
fi

# ------------------------------------- 1a. evaluation sets (CPU, concurrent)
# Cheap, and the earliest hard failure point; also enables the V5-vs-V5 dry-run
# validation while synthesis is still running. Deliberately few workers so it
# does not starve synthesis (24) or the concurrent GUNC job (~8).
NG=$($PY -c "import json;print(len(json.load(open('data/holdout_family/eval_sets/manifest.json'))['groups']))" 2>/dev/null || echo 0)
if [ "$NG" -lt 7 ]; then
  say "generating evaluation sets (123), concurrent with synthesis"
  $PY scripts/123_generate_holdout_eval_sets.py --workers 8 \
      >> logs/revision/ws1.9_full_123.out 2>&1
fi
say "evaluation sets ready ($($PY -c "import json;print(len(json.load(open('data/holdout_family/eval_sets/manifest.json'))['groups']))" 2>/dev/null || echo 0) groups)"

# ------------------------------------------ 1b. DRY-RUN DESIGN VALIDATION
# Substituting production V5 for BOTH models must give exactly zero for every
# difference. WS1.6 did this and it passed; repeated here on the family sets.
if [ ! -f results/revision/holdout_family/dryrun_v5_vs_v5/lineage_novelty_effect_did.tsv ]; then
  say "dry-run validation: production V5 substituted for both models"
  $PY scripts/124_evaluate_holdout.py \
      --holdout-onnx models/magicc_v5.onnx \
      --holdout-norm data/features/normalization_params.json \
      --out-dir results/revision/holdout_family/dryrun_v5_vs_v5 \
      >> logs/revision/ws1.9_dryrun.log 2>&1
fi
$PY - <<'EOF' 2>&1 | tee -a "$LOG"
import pandas as pd, pathlib
p = pathlib.Path('results/revision/holdout_family/dryrun_v5_vs_v5/'
                 'lineage_novelty_effect_did.tsv')
if p.exists():
    d = pd.read_csv(p, sep='\t')
    cols = [c for c in d.columns if c.endswith(('_did', '_delta'))]
    mx = float(d[cols].abs().to_numpy().max())
    print(f'DRY-RUN VALIDATION: max |difference| over {cols} = {mx:.3e} '
          f'-> {"PASS (exactly zero)" if mx == 0.0 else "FAIL"}')
EOF

# ------------------------------------------------- 2. wait for synthesis (CPU)
say "waiting for synthesis (121): 120 batches + normalization, ~7 h"
while true; do
  NORM=$($PY - <<'EOF' 2>/dev/null || echo no
import json
d = json.load(open('data/holdout_family/build_checkpoint.json'))
print('yes' if d.get('phases', {}).get('normalized') == 'complete' else 'no')
EOF
)
  [ "$NORM" = "yes" ] && break
  # restart synthesis if it died (it resumes from its per-batch checkpoint)
  if ! pgrep -f "121_build_holdout_training_data" > /dev/null; then
    say "synthesis not running - (re)starting; resumes from checkpoint"
    MAGICC_HOLDOUT_LEVEL=family PYTHONHASHSEED=0 \
      setsid $PY scripts/121_build_holdout_training_data.py --workers $WORKERS \
        >> logs/revision/ws1.9_full_121.out 2>&1 < /dev/null &
    disown
    sleep 30
  fi
  sleep 120
done
say "synthesis complete and normalized"

# ---------------------------------------------------------- 3. training (GPU)
if [ ! -f models/magicc_holdout_family.onnx ]; then
  say "starting/resuming holdout training (122) - expect ~6 h; seed 42"
  for ATTEMPT in 1 2 3; do
    RESUME=""
    ls models/holdout_family/checkpoint_epoch_*.pt > /dev/null 2>&1 && RESUME="--resume-auto"
    $PY scripts/122_train_holdout_phylum.py --seed 42 $RESUME \
        >> logs/revision/ws1.9_train_holdout.log 2>&1
    [ -f models/magicc_holdout_family.onnx ] && break
    say "training attempt $ATTEMPT did not produce the ONNX; retrying from checkpoint"
    sleep 60
  done
fi
[ -f models/magicc_holdout_family.onnx ] || { say "TRAINING FAILED - no ONNX"; exit 1; }
say "training + ONNX export complete"

# -------------------------------------------------------------- 4. evaluation
say "evaluating holdout vs production V5 (124) on identical samples"
$PY scripts/124_evaluate_holdout.py >> logs/revision/ws1.9_evaluate.log 2>&1 \
  || { say "EVALUATION FAILED"; exit 1; }
say "evaluation complete"

# ------------------------------------------- 5. family vs phylum comparison
say "family-vs-phylum comparison (129)"
$PY scripts/129_family_vs_phylum_comparison.py \
    >> logs/revision/ws1.9_comparison.log 2>&1 \
  || say "comparison step failed (non-fatal; evaluation outputs are already written)"

say "PIPELINE COMPLETE - results in results/revision/holdout_family/"
