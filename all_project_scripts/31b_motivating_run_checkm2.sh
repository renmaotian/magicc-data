#!/bin/bash
# Run CheckM2 on motivating benchmark sets A and B
# CheckM2 in conda env checkm2_py39, 32 threads
# Includes --dbg_vectors for DeepCheck compatibility

set -euo pipefail

PROJECT_DIR="/home/tianrm/projects/magicc2"
CHECKM2_DB="${PROJECT_DIR}/tools/checkm2_db/CheckM2_database/uniref100.KO.1.dmnd"
CONDA_ENV="checkm2_py39"
THREADS=32
BENCHMARK_DIR="${PROJECT_DIR}/data/benchmarks/motivating"

echo "=============================================="
echo "CheckM2 Motivating Benchmark Run - $(date)"
echo "=============================================="
echo "Threads: ${THREADS}"
echo "Database: ${CHECKM2_DB}"
echo ""

TOTAL_START=$(date +%s)

for SET in A B; do
    FASTA_DIR="${BENCHMARK_DIR}/set_${SET}/fasta"
    OUTPUT_DIR="${BENCHMARK_DIR}/set_${SET}/checkm2_output"
    QUALITY_REPORT="${OUTPUT_DIR}/quality_report.tsv"
    TIME_FILE="${BENCHMARK_DIR}/set_${SET}/checkm2_wallclock.txt"

    echo "----------------------------------------------"
    echo "Set ${SET}: ${FASTA_DIR}"

    N_GENOMES=$(ls "${FASTA_DIR}"/*.fasta 2>/dev/null | wc -l)
    echo "  Genomes: ${N_GENOMES}"

    # Check if already completed
    if [ -f "${QUALITY_REPORT}" ]; then
        N_RESULTS=$(tail -n +2 "${QUALITY_REPORT}" | wc -l)
        if [ "${N_RESULTS}" -ge "${N_GENOMES}" ]; then
            echo "  SKIPPING: Output already exists with ${N_RESULTS} results (>= ${N_GENOMES} genomes)"
            continue
        else
            echo "  Incomplete output (${N_RESULTS}/${N_GENOMES}). Re-running with --force."
        fi
    fi

    echo "  Starting CheckM2 at $(date)"
    SET_START=$(date +%s)

    conda run -n "${CONDA_ENV}" env CHECKM2DB="${CHECKM2_DB}" \
        checkm2 predict \
        --threads "${THREADS}" \
        -x .fasta \
        --input "${FASTA_DIR}" \
        --output-directory "${OUTPUT_DIR}" \
        --force \
        --dbg_vectors \
        2>&1

    SET_END=$(date +%s)
    SET_ELAPSED=$((SET_END - SET_START))

    echo "${SET_ELAPSED}" > "${TIME_FILE}"

    echo "  Finished Set ${SET} at $(date)"
    echo "  Wall-clock time: ${SET_ELAPSED} seconds ($(echo "scale=1; ${SET_ELAPSED}/60" | bc) minutes)"

    if [ -f "${QUALITY_REPORT}" ]; then
        N_RESULTS=$(tail -n +2 "${QUALITY_REPORT}" | wc -l)
        echo "  Results: ${N_RESULTS} genomes in quality_report.tsv"
    else
        echo "  WARNING: quality_report.tsv not found!"
    fi

    echo "  Output size: $(du -sh "${OUTPUT_DIR}" | cut -f1)"
    echo ""
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo "=============================================="
echo "All motivating sets completed at $(date)"
echo "Total wall-clock time: ${TOTAL_ELAPSED} seconds ($(echo "scale=1; ${TOTAL_ELAPSED}/60" | bc) minutes)"
echo "=============================================="
