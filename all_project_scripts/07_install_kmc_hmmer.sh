#!/bin/bash
# 07_install_kmc_hmmer.sh - Verify KMC3, HMMER, and Prodigal installation in magicc2 env
# All tools were already installed; this script verifies functionality

set -euo pipefail

echo "=== Verifying tool installations in magicc2 conda environment ==="

echo ""
echo "--- HMMER ---"
hmmsearch -h 2>&1 | head -2

echo ""
echo "--- KMC3 ---"
kmc 2>&1 | head -1

echo ""
echo "--- KMC tools ---"
kmc_tools 2>&1 | head -1
kmc_dump 2>&1 | head -1

echo ""
echo "--- Prodigal ---"
prodigal -v 2>&1

echo ""
echo "--- HMM Profile Files ---"
BCG_HMM="/home/tianrm/projects/magicc2/85_bcg.hmm"
UACG_HMM="/home/tianrm/projects/magicc2/uacg.hmm"

echo "85_bcg.hmm: $(wc -l < $BCG_HMM) lines, $(grep -c '^NAME ' $BCG_HMM) models"
echo "uacg.hmm: $(wc -l < $UACG_HMM) lines, $(grep -c '^NAME ' $UACG_HMM) models"

echo ""
echo "--- Quick functional test: hmmsearch with 85_bcg.hmm ---"
# Pick a random genome for testing
TEST_GENOME=$(head -2 /home/tianrm/projects/magicc2/data/splits/train_genomes.tsv | tail -1 | cut -f13)
echo "Test genome: $TEST_GENOME"

TMPDIR=$(mktemp -d)
# Run prodigal to get proteins
prodigal -i "$TEST_GENOME" -a "${TMPDIR}/test_proteins.faa" -o "${TMPDIR}/test_genes.gff" -f gff -p single -q 2>/dev/null
echo "Prodigal: $(grep -c '^>' ${TMPDIR}/test_proteins.faa) proteins predicted"

# Run hmmsearch
hmmsearch --cut_tc --tblout "${TMPDIR}/test_hits.tbl" --cpu 4 --noali "$BCG_HMM" "${TMPDIR}/test_proteins.faa" > /dev/null
HITS=$(grep -v '^#' "${TMPDIR}/test_hits.tbl" | wc -l)
echo "hmmsearch: $HITS hits found with --cut_tc"

# Quick KMC test
echo "$TEST_GENOME" > "${TMPDIR}/input.txt"
kmc -k9 -ci1 -cs65535 -fm "${TMPDIR}/input.txt" "${TMPDIR}/kmc_test" "${TMPDIR}" > /dev/null 2>&1
kmc_dump "${TMPDIR}/kmc_test" "${TMPDIR}/kmc_dump.txt" > /dev/null 2>&1
KMER_COUNT=$(wc -l < "${TMPDIR}/kmc_dump.txt")
echo "KMC3: $KMER_COUNT unique canonical 9-mers counted"

rm -rf "$TMPDIR"

echo ""
echo "=== All tools verified successfully ==="
