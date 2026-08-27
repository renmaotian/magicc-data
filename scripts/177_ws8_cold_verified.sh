#!/bin/bash
# WS8.2 CORRECTION (2026-07-31) — re-run the cold-cache arm with a page-cache
# eviction that provably works.
#
# The 92-cell campaign's `cache=cold` cells did not evict anything: script 162
# raised TypeError on its first file and died before any posix_fadvise call
# (see runs/*.evict.txt, all identical tracebacks) and every cold run recorded
# `File system inputs = 0`. Those cells are warm runs with a cold label and are
# reported as such. This script repeats the cold arm using scripts/176, which
# verifies with mincore(2) that residency went to zero, and labels the cells
# `cold_verified` so the invalid and valid cells never mix.
#
# Scope is deliberately small: it re-establishes the cold-vs-warm delta for
# every tool at one thread count each, not the whole factorial.
#
#   setsid nohup bash scripts/177_ws8_cold_verified.sh > logs/revision/ws8_cold_verified.log 2>&1 &
set -uo pipefail

PROJECT="/path/to/magicc"
QUIET_LOAD="${QUIET_LOAD:-1.0}"
QUIET_MAX_WAIT="${QUIET_MAX_WAIT:-900}"

wait_for_quiet() {
    local waited=0 l1
    while : ; do
        l1=$(cut -d' ' -f1 /proc/loadavg)
        if awk -v a="${l1}" -v b="${QUIET_LOAD}" 'BEGIN{exit !(a<b)}'; then return 0; fi
        if [ "${waited}" -ge "${QUIET_MAX_WAIT}" ]; then
            echo "[warn] load ${l1} still >= ${QUIET_LOAD} after ${waited}s; proceeding and recording it"
            return 1
        fi
        sleep 15; waited=$((waited + 15))
    done
}

# tool threads repeat inputset
PLAN=(
  "magicc 1 1 set_E_full"
  "magicc 32 1 set_E_full"
  "magicc 1 2 set_E_full"
  "magicc 32 2 set_E_full"
  "magicc 1 3 set_E_full"
  "magicc 32 3 set_E_full"
  "deepcheck 1 1 set_E_100"
  "deepcheck 1 2 set_E_100"
  "cocopye 32 1 set_E_100"
  "checkm2 32 1 set_E_100"
)

echo "WS8 cold_verified start $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"
for row in "${PLAN[@]}"; do
    read -r TOOL THREADS REP INPUTSET <<< "${row}"
    wait_for_quiet
    echo "[run ] ${TOOL} t${THREADS} r${REP} ${INPUTSET} load=$(cut -d' ' -f1-3 /proc/loadavg) $(date -u +%H:%M:%SZ)"
    bash "${PROJECT}/scripts/161_ws8_run_one.sh" "${TOOL}" "${THREADS}" "${REP}" "${INPUTSET}" cold_verified
done
echo "WS8 cold_verified end $(date -u +%Y-%m-%dT%H:%M:%SZ)"
