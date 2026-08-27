#!/bin/bash
# =============================================================================
# WS2.3 verification -- resumability and determinism of the Set F generator.
#
# Deletes a handful of generated FASTAs (and their checkpoint lines), re-runs
# scripts/155_generate_set_F.py, and asserts that the regenerated files are
# BYTE-IDENTICAL to the originals and that metadata.tsv / generation_metadata.tsv
# are unchanged.  Same protocol as the set_C_clean / set_D_clean check.
#
# Usage: bash scripts/155b_setF_determinism_check.sh [set_F] [n_samples]
# =============================================================================
set -uo pipefail
PROJECT_DIR="/path/to/magicc"
SET_NAME="${1:-set_F}"
N="${2:-6}"
SET_DIR="${PROJECT_DIR}/data/benchmarks/${SET_NAME}"
PY=/path/to/anaconda3/envs/magicc2/bin/python
WORK=$(mktemp -d)
trap 'rm -rf "${WORK}"' EXIT

echo "=== Set F determinism / resumability check (${SET_NAME}, ${N} samples) ==="
cd "${PROJECT_DIR}"

# pick N samples spanning all three contamination types
mapfile -t GIDS < <("${PY}" - "${SET_DIR}" "${N}" <<'PY'
import sys, pandas as pd
d = pd.read_csv(f"{sys.argv[1]}/generation_metadata.tsv", sep="\t")
n = int(sys.argv[2])
out = []
for t, sub in d.groupby("contamination_type"):
    out += list(sub.sort_values("genome_id").genome_id.head(max(1, n // max(1, d.contamination_type.nunique()))))
print("\n".join(out[:n]))
PY
)
echo "sampled: ${GIDS[*]}"

cp "${SET_DIR}/metadata.tsv" "${WORK}/metadata.before.tsv"
cp "${SET_DIR}/generation_metadata.tsv" "${WORK}/genmeta.before.tsv"
for g in "${GIDS[@]}"; do
    cp "${SET_DIR}/fasta/${g}.fasta" "${WORK}/${g}.before.fasta"
    rm -f "${SET_DIR}/fasta/${g}.fasta"
done

# drop their checkpoint lines so the generator must rebuild them
"${PY}" - "${SET_DIR}/generation_checkpoint.jsonl" "${GIDS[@]}" <<'PY'
import json, sys
path, gids = sys.argv[1], set(sys.argv[2:])
keep = []
with open(path) as fh:
    for line in fh:
        try:
            if json.loads(line).get("genome_id") in gids:
                continue
        except json.JSONDecodeError:
            pass
        keep.append(line)
open(path, "w").writelines(keep)
print(f"checkpoint: {len(keep)} lines kept, {len(gids)} removed")
PY

echo "--- regenerating ---"
EXTRA=""
[[ "${SET_NAME}" == "set_F_pilot" ]] && EXTRA="--pilot"
"${PY}" scripts/155_generate_set_F.py ${EXTRA} --workers 6 2>&1 | tail -6

FAIL=0
for g in "${GIDS[@]}"; do
    if cmp -s "${WORK}/${g}.before.fasta" "${SET_DIR}/fasta/${g}.fasta"; then
        echo "  [PASS] ${g} byte-identical"
    else
        echo "  [FAIL] ${g} DIFFERS"; FAIL=1
    fi
done
for f in metadata genmeta; do
    src="${SET_DIR}/metadata.tsv"; [[ "$f" == genmeta ]] && src="${SET_DIR}/generation_metadata.tsv"
    if cmp -s "${WORK}/${f}.before.tsv" "${src}"; then
        echo "  [PASS] $(basename "${src}") unchanged"
    else
        echo "  [FAIL] $(basename "${src}") CHANGED"; FAIL=1
    fi
done
echo "DETERMINISM CHECK: $([[ ${FAIL} -eq 0 ]] && echo PASS || echo FAIL)"
exit ${FAIL}
