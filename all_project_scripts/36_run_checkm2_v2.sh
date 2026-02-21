#!/bin/bash
# Script 36: Run CheckM2 on all 5 new finished-genome benchmark sets
# Sets: motivating_v2/set_A, motivating_v2/set_B, set_A_v2, set_B_v2, set_E
# Each set has 1,000 genomes
# CheckM2 installed in conda env checkm2_py39 (Python 3.9.23)
# Database: /home/tianrm/projects/magicc2/tools/checkm2_db/CheckM2_database/uniref100.KO.1.dmnd
# Uses 32 threads, runs sets sequentially, resumable (skips existing output)

set -euo pipefail

PROJECT_DIR="/home/tianrm/projects/magicc2"
CHECKM2_DB="${PROJECT_DIR}/tools/checkm2_db/CheckM2_database/uniref100.KO.1.dmnd"
CONDA_ENV="checkm2_py39"
THREADS=32
BENCHMARK_DIR="${PROJECT_DIR}/data/benchmarks"
LOG_DIR="${PROJECT_DIR}/logs"

mkdir -p "${LOG_DIR}"

echo "=============================================="
echo "CheckM2 Benchmark Run (v2 finished-genome sets) - $(date)"
echo "=============================================="
echo "Threads: ${THREADS}"
echo "Database: ${CHECKM2_DB}"
echo "Conda env: ${CONDA_ENV}"
echo ""

# Define the 5 sets: label|fasta_dir|output_dir
SETS=(
    "motivating_v2_set_A|${BENCHMARK_DIR}/motivating_v2/set_A/fasta|${BENCHMARK_DIR}/motivating_v2/set_A/checkm2_output|${BENCHMARK_DIR}/motivating_v2/set_A"
    "motivating_v2_set_B|${BENCHMARK_DIR}/motivating_v2/set_B/fasta|${BENCHMARK_DIR}/motivating_v2/set_B/checkm2_output|${BENCHMARK_DIR}/motivating_v2/set_B"
    "set_A_v2|${BENCHMARK_DIR}/set_A_v2/fasta|${BENCHMARK_DIR}/set_A_v2/checkm2_output|${BENCHMARK_DIR}/set_A_v2"
    "set_B_v2|${BENCHMARK_DIR}/set_B_v2/fasta|${BENCHMARK_DIR}/set_B_v2/checkm2_output|${BENCHMARK_DIR}/set_B_v2"
    "set_E|${BENCHMARK_DIR}/set_E/fasta|${BENCHMARK_DIR}/set_E/checkm2_output|${BENCHMARK_DIR}/set_E"
)

# Track total time
TOTAL_START=$(date +%s)
SETS_COMPLETED=0

for SET_ENTRY in "${SETS[@]}"; do
    IFS='|' read -r SET_LABEL FASTA_DIR OUTPUT_DIR SET_DIR <<< "${SET_ENTRY}"

    QUALITY_REPORT="${OUTPUT_DIR}/quality_report.tsv"
    LOG_FILE="${LOG_DIR}/checkm2_${SET_LABEL}.log"
    TIME_FILE="${SET_DIR}/checkm2_wallclock.txt"

    echo "----------------------------------------------"
    echo "${SET_LABEL}: ${FASTA_DIR}"

    # Count genomes
    N_GENOMES=$(ls "${FASTA_DIR}"/*.fasta 2>/dev/null | wc -l)
    echo "  Genomes: ${N_GENOMES}"

    # Check if already completed
    if [ -f "${QUALITY_REPORT}" ]; then
        N_RESULTS=$(tail -n +2 "${QUALITY_REPORT}" | wc -l)
        if [ "${N_RESULTS}" -ge "${N_GENOMES}" ]; then
            echo "  SKIPPING: Output already exists with ${N_RESULTS} results (>= ${N_GENOMES} genomes)"
            # Read existing wall-clock time if available
            if [ -f "${TIME_FILE}" ]; then
                echo "  Existing wall-clock: $(cat "${TIME_FILE}")s"
            fi
            SETS_COMPLETED=$((SETS_COMPLETED + 1))
            continue
        else
            echo "  Incomplete output (${N_RESULTS}/${N_GENOMES}). Re-running with --force."
        fi
    fi

    echo "  Starting CheckM2 at $(date)"
    SET_START=$(date +%s)

    # Run CheckM2 with --dbg_vectors for DeepCheck compatibility
    conda run -n "${CONDA_ENV}" env CHECKM2DB="${CHECKM2_DB}" \
        checkm2 predict \
        --threads "${THREADS}" \
        -x .fasta \
        --input "${FASTA_DIR}" \
        --output-directory "${OUTPUT_DIR}" \
        --force \
        --dbg_vectors \
        2>&1 | tee "${LOG_FILE}"

    SET_END=$(date +%s)
    SET_ELAPSED=$((SET_END - SET_START))

    echo "${SET_ELAPSED}" > "${TIME_FILE}"

    echo "  Finished ${SET_LABEL} at $(date)"
    echo "  Wall-clock time: ${SET_ELAPSED} seconds ($(echo "scale=1; ${SET_ELAPSED}/60" | bc) minutes)"

    # Verify output
    if [ -f "${QUALITY_REPORT}" ]; then
        N_RESULTS=$(tail -n +2 "${QUALITY_REPORT}" | wc -l)
        echo "  Results: ${N_RESULTS} genomes in quality_report.tsv"
    else
        echo "  WARNING: quality_report.tsv not found!"
    fi

    # Check disk usage
    echo "  Output size: $(du -sh "${OUTPUT_DIR}" | cut -f1)"
    echo ""

    SETS_COMPLETED=$((SETS_COMPLETED + 1))
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo "=============================================="
echo "All ${SETS_COMPLETED}/5 sets completed at $(date)"
echo "Total wall-clock time: ${TOTAL_ELAPSED} seconds ($(echo "scale=1; ${TOTAL_ELAPSED}/60" | bc) minutes)"
echo "=============================================="
