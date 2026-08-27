#!/bin/bash
# WS11.T — execute results/revision/speed_v033/run_plan_v033.tsv, SERIALLY.
#
# Collapse-safe and resumable: every cell writes its own JSON and is skipped if
# that JSON already exists, so re-launching after any interruption continues
# where it stopped.  Nothing is ever run concurrently.
#
# IDLENESS RULE (as amended by the coordinator 2026-08-26):
#   * host state (all three load averages, nproc, MemAvailable, vmstat) is
#     recorded immediately before AND after every timed cell, by 191;
#   * a general cell starts when the pre-run 1-minute load average is < 4.0
#     (on a 48-core host a 1-2 core editorial background load does not contend
#     with a 32-thread run);
#   * a MAGICC 1-THREAD cell, where one competing core matters proportionally,
#     requires < 1.5 and polls every 60 s for up to 20 minutes; if it is still
#     above 1.5 the run proceeds and is FLAGGED rather than stalling;
#   * no completed run is ever dropped.  Flagging and extra repeats are handled
#     downstream by scripts/195.
#
#   setsid nohup bash scripts/254_ws11t_campaign.sh \
#       > results/revision/speed_v033/logs/campaign.log 2>&1 &
set -uo pipefail

PROJECT="/path/to/magicc"
SPEED="${PROJECT}/results/revision/speed_v033"
PLAN="${SPEED}/run_plan_v033.tsv"
GATE_GENERAL="${GATE_GENERAL:-4.0}"
GATE_MAGICC1="${GATE_MAGICC1:-1.5}"
MAGICC1_MAX_WAIT="${MAGICC1_MAX_WAIT:-1200}"   # 20 min
GENERAL_MAX_WAIT="${GENERAL_MAX_WAIT:-900}"    # 15 min
POLL_S="${POLL_S:-30}"
MAGICC1_POLL_S="${MAGICC1_POLL_S:-60}"
MIN_TIER="${MIN_TIER:-0}"
MAX_TIER="${MAX_TIER:-9}"

mkdir -p "${SPEED}/logs"
GATELOG="${SPEED}/logs/load_gate.tsv"
if [ ! -s "${GATELOG}" ]; then
    printf 'utc\tcell\tgate\tthreshold\tload1_at_decision\twaited_s\tdecision\tbusy_pct_all_procs\n' > "${GATELOG}"
fi

# The 1-minute load average has a ~60 s decay constant, so immediately after
# one of our OWN 32-thread cells it reads 20-40 even on a completely idle host.
# Gating on that raw figure would flag every cell that follows an expensive one.
# The gate therefore lets the machine SETTLE: it polls until the 1-minute load
# average falls below the threshold, up to a cap, then runs anyway and flags.
# `busy` is the instantaneous sum of %CPU over all processes (`ps -eo pcpu`),
# which -- unlike the load average -- carries no memory of our own last run and
# is the honest measure of EXTERNAL contention.  It is recorded either way.
gate() {
    local cell="$1" thresh="$2" maxwait="$3" poll="$4" waited=0 l1 busy
    while : ; do
        l1=$(cut -d' ' -f1 /proc/loadavg)
        busy=$(ps -eo pcpu= | awk '{s+=$1} END {printf "%.1f", s}')
        if awk -v a="${l1}" -v b="${thresh}" 'BEGIN{exit !(a<b)}'; then
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${cell}" "load1" "${thresh}" \
                "${l1}" "${waited}" "PROCEED" "${busy}" >> "${GATELOG}"
            return 0
        fi
        if [ "${waited}" -ge "${maxwait}" ]; then
            echo "[warn] load ${l1} still >= ${thresh} after ${waited}s (busy=${busy}%); running anyway and FLAGGING"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${cell}" "load1" "${thresh}" \
                "${l1}" "${waited}" "PROCEED_FLAGGED_OVER_THRESHOLD" "${busy}" >> "${GATELOG}"
            return 1
        fi
        echo "[wait] ${cell}: load1=${l1} >= ${thresh} (busy=${busy}%), settling, waited ${waited}s"
        sleep "${poll}"
        waited=$((waited + poll))
    done
}

echo "=================================================================="
echo "WS11.T v0.3.3 campaign start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "host=$(hostname) nproc=$(nproc) plan=${PLAN}"
echo "tiers ${MIN_TIER}..${MAX_TIER}  gate_general=${GATE_GENERAL} gate_magicc1thr=${GATE_MAGICC1}"
echo "mem: $(free -g | awk '/^Mem:/{print $2" GiB total, "$7" GiB available"}')"
echo "=================================================================="

while IFS=$'\t' read -r TIER ORDER TOOL THREADS REP INPUTSET CACHE NOTE; do
    [ -z "${TOOL:-}" ] && continue
    [ "${TIER}" = "tier" ] && continue
    if [ "${TIER}" -gt "${MAX_TIER}" ] || [ "${TIER}" -lt "${MIN_TIER}" ]; then continue; fi

    CELL="${TOOL}__t${THREADS}__${INPUTSET}__${CACHE}__r${REP}"
    if [ -s "${SPEED}/runs/${CELL}.json" ] && grep -q '"wall_clock_s"' "${SPEED}/runs/${CELL}.json" 2>/dev/null; then
        echo "[skip] ${ORDER}/${CELL}"
        continue
    fi

    case "${TOOL}" in
      magicc*) if [ "${THREADS}" -eq 1 ]; then
                   gate "${CELL}" "${GATE_MAGICC1}" "${MAGICC1_MAX_WAIT}" "${MAGICC1_POLL_S}"
               else
                   gate "${CELL}" "${GATE_GENERAL}" "${GENERAL_MAX_WAIT}" "${POLL_S}"
               fi ;;
      *)       gate "${CELL}" "${GATE_GENERAL}" "${GENERAL_MAX_WAIT}" "${POLL_S}" ;;
    esac

    echo "[run ] tier=${TIER} order=${ORDER} ${CELL} | ${NOTE}"
    echo "       loadavg=$(cut -d' ' -f1-3 /proc/loadavg) memavail=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)kB $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    bash "${PROJECT}/scripts/259_ws11t_v033_run_one.sh" "${TOOL}" "${THREADS}" "${REP}" "${INPUTSET}" "${CACHE}"
done < <(tail -n +2 "${PLAN}")

echo "=================================================================="
echo "WS11.T v0.3.3 campaign end $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=================================================================="
