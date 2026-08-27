#!/bin/bash
# WS8.1 — execute the speed campaign from results/revision/speed/run_plan.tsv.
#
# Collapse-safe and resumable: every cell writes its own JSON and is skipped if
# that JSON already exists, so re-launching after any interruption continues.
#
# Before every timed run the driver WAITS for the machine to go quiet
# (1-minute load average below QUIET_LOAD) and records the load average that
# was actually in force. A contended measurement is therefore visible in the
# record rather than silently reported.
#
#   setsid nohup bash scripts/166_ws8_campaign.sh > logs/revision/ws8_campaign.log 2>&1 &
set -uo pipefail

PROJECT="/path/to/magicc"
SPEED="${PROJECT}/results/revision/speed"
PLAN="${SPEED}/run_plan.tsv"
QUIET_LOAD="${QUIET_LOAD:-1.0}"
QUIET_MAX_WAIT="${QUIET_MAX_WAIT:-900}"
MAX_TIER="${MAX_TIER:-9}"

wait_for_quiet() {
    local waited=0
    while : ; do
        local l1
        l1=$(cut -d' ' -f1 /proc/loadavg)
        if awk -v a="${l1}" -v b="${QUIET_LOAD}" 'BEGIN{exit !(a<b)}'; then
            return 0
        fi
        if [ "${waited}" -ge "${QUIET_MAX_WAIT}" ]; then
            echo "[warn] load ${l1} still >= ${QUIET_LOAD} after ${waited}s; proceeding and recording it"
            return 1
        fi
        sleep 15
        waited=$((waited + 15))
    done
}

echo "=================================================================="
echo "WS8 speed campaign start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "host=$(hostname) nproc=$(nproc) plan=${PLAN} MAX_TIER=${MAX_TIER}"
echo "=================================================================="

tail -n +2 "${PLAN}" | while IFS=$'\t' read -r TIER ORDER TOOL THREADS REP INPUTSET CACHE; do
    [ -z "${TOOL:-}" ] && continue
    if [ "${TIER}" -gt "${MAX_TIER}" ]; then continue; fi

    CELL="${TOOL}__t${THREADS}__${INPUTSET}__${CACHE}__r${REP}"
    if [ -s "${SPEED}/runs/${CELL}.json" ] && grep -q '"wall_clock_s"' "${SPEED}/runs/${CELL}.json" 2>/dev/null; then
        echo "[skip] ${ORDER}/${CELL}"
        continue
    fi

    wait_for_quiet
    echo "[run ] tier=${TIER} order=${ORDER} ${CELL} load=$(cut -d' ' -f1-3 /proc/loadavg) $(date -u +%H:%M:%SZ)"
    bash "${PROJECT}/scripts/161_ws8_run_one.sh" "${TOOL}" "${THREADS}" "${REP}" "${INPUTSET}" "${CACHE}"
done

echo "=================================================================="
echo "WS8 speed campaign end $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=================================================================="
