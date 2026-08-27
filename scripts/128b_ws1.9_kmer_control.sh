#!/usr/bin/env bash
# WS1.9 addendum - k-mer feature-selection leakage control at FAMILY level
# (the WS1.7 analogue). Run as a SEPARATE detached job so the main orchestrator
# (128) is never edited while bash is executing it.
#
# Waits until 128 has finished (the DiD table exists), then reselects the 9-mer
# set from feature-selection representatives with the panel FAMILIES removed and
# compares it to the production set.
#
#   setsid nohup bash scripts/128b_ws1.9_kmer_control.sh \
#       >> logs/revision/ws1.9_kmer_control.out 2>&1 < /dev/null &
#
set -u
cd /path/to/magicc
export MAGICC_HOLDOUT_LEVEL=family
export PYTHONHASHSEED=0
PY=/path/to/conda/envs/magicc2/bin/python
LOG=logs/revision/ws1.9_pipeline.log
say() { echo "[$(date '+%F %T')] [128b] $*" | tee -a "$LOG"; }

say "waiting for the main pipeline to finish before starting the k-mer control"
while [ ! -f results/revision/holdout_family/lineage_novelty_effect_did.tsv ]; do
  sleep 300
done
sleep 120     # let 129 finish too

if [ -f results/revision/holdout_family/kmer_reselection_summary.json ]; then
  say "k-mer control already done"; exit 0
fi
say "running k-mer reselection control (125) at family level, 24 workers"
$PY scripts/125_kmer_reselection_control.py --workers 24 \
    > logs/revision/ws1.9_kmer_reselection.log 2>&1 \
  && say "k-mer reselection control COMPLETE" \
  || say "k-mer reselection control FAILED (non-fatal; the WS1.7 phylum-level \
control already covers this and the family panel touches 3x fewer feature-\
selection representatives)"
