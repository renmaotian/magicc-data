#!/bin/bash
# Step 3: Download genomes using NCBI datasets CLI
# Uses batching (200 accessions per batch) with 10 parallel workers
# Collapse-safe: checks for existing files before re-downloading
# Each batch downloads to a temp zip, extracts, and moves genome dirs to final location.

set -euo pipefail

ACC_FILE="/home/tianrm/projects/magicc2/data/gtdb/selected_100k_accessions.txt"
GENOME_DIR="/home/tianrm/projects/magicc2/data/genomes"
BATCH_DIR="/home/tianrm/projects/magicc2/data/genome_batches"
LOG_DIR="/home/tianrm/projects/magicc2/data/download_logs"
BATCH_SIZE=200
NUM_WORKERS=10

mkdir -p "$GENOME_DIR" "$BATCH_DIR" "$LOG_DIR"

# Count total accessions
TOTAL=$(wc -l < "$ACC_FILE")
echo "Total accessions to download: $TOTAL"
echo "Batch size: $BATCH_SIZE"

# Split accessions into batch files
echo "Splitting accessions into batches of $BATCH_SIZE..."
split -l $BATCH_SIZE -d -a 4 "$ACC_FILE" "$BATCH_DIR/batch_"

NUM_BATCHES=$(ls "$BATCH_DIR"/batch_* | wc -l)
echo "Created $NUM_BATCHES batch files"

# Function to download a single batch
download_batch() {
    local BATCH_FILE="$1"
    local BATCH_NAME=$(basename "$BATCH_FILE")
    local BATCH_LOG="$LOG_DIR/${BATCH_NAME}.log"
    local DONE_MARKER="$LOG_DIR/${BATCH_NAME}.done"
    local FAIL_MARKER="$LOG_DIR/${BATCH_NAME}.failed"
    local GENOME_DIR="/home/tianrm/projects/magicc2/data/genomes"

    # Skip if already done
    if [ -f "$DONE_MARKER" ]; then
        return 0
    fi

    # Remove fail marker if retrying
    rm -f "$FAIL_MARKER"

    local NUM_ACC=$(wc -l < "$BATCH_FILE")

    # Download using datasets to a temp file in /tmp (not inside GENOME_DIR)
    local TMPZIP="/tmp/ncbi_dl_${BATCH_NAME}_$$.zip"
    local TMPEXTRACT="/tmp/ncbi_extract_${BATCH_NAME}_$$"

    # Retry up to 3 times
    local MAX_RETRIES=3
    local RETRY=0
    local SUCCESS=0

    while [ $RETRY -lt $MAX_RETRIES ] && [ $SUCCESS -eq 0 ]; do
        RETRY=$((RETRY + 1))
        rm -f "$TMPZIP"
        rm -rf "$TMPEXTRACT"

        if datasets download genome accession \
            --inputfile "$BATCH_FILE" \
            --include genome \
            --filename "$TMPZIP" \
            --no-progressbar \
            >> "$BATCH_LOG" 2>&1; then

            # Verify zip is valid
            if [ -f "$TMPZIP" ] && unzip -t "$TMPZIP" > /dev/null 2>&1; then
                # Extract
                mkdir -p "$TMPEXTRACT"
                if unzip -q -o "$TMPZIP" -d "$TMPEXTRACT" >> "$BATCH_LOG" 2>&1; then
                    # Move genome directories
                    local MOVED=0
                    if [ -d "$TMPEXTRACT/ncbi_dataset/data" ]; then
                        for acc_dir in "$TMPEXTRACT"/ncbi_dataset/data/GC*; do
                            if [ -d "$acc_dir" ]; then
                                local acc=$(basename "$acc_dir")
                                local target_dir="$GENOME_DIR/$acc"
                                if [ ! -d "$target_dir" ]; then
                                    mv "$acc_dir" "$target_dir" 2>/dev/null && MOVED=$((MOVED + 1)) || true
                                else
                                    MOVED=$((MOVED + 1))
                                fi
                            fi
                        done
                    fi
                    SUCCESS=1
                    echo "[DONE] $BATCH_NAME: $MOVED genomes (attempt $RETRY)" >> "$BATCH_LOG"
                fi
            else
                echo "[RETRY] $BATCH_NAME: invalid zip (attempt $RETRY)" >> "$BATCH_LOG"
                sleep $((RETRY * 5))
            fi
        else
            echo "[RETRY] $BATCH_NAME: download failed (attempt $RETRY)" >> "$BATCH_LOG"
            sleep $((RETRY * 5))
        fi
    done

    # Cleanup
    rm -f "$TMPZIP"
    rm -rf "$TMPEXTRACT"

    if [ $SUCCESS -eq 1 ]; then
        touch "$DONE_MARKER"
    else
        touch "$FAIL_MARKER"
        echo "[FAIL] $BATCH_NAME after $MAX_RETRIES retries" >> "$BATCH_LOG"
        return 1
    fi
}

export -f download_batch

# Run downloads with parallel workers
echo ""
echo "Starting download with $NUM_WORKERS parallel workers..."
echo "Progress will be logged to $LOG_DIR/"
echo ""

ls "$BATCH_DIR"/batch_* | xargs -P $NUM_WORKERS -I {} bash -c 'download_batch "$@"' _ {}

# Report results
DONE_COUNT=$(ls "$LOG_DIR"/*.done 2>/dev/null | wc -l)
FAIL_COUNT=$(ls "$LOG_DIR"/*.failed 2>/dev/null | wc -l)
GENOME_COUNT=$(find "$GENOME_DIR" -maxdepth 1 -type d -name "GC*" | wc -l)

echo ""
echo "========================================="
echo "Download Summary"
echo "========================================="
echo "  Batches completed: $DONE_COUNT / $NUM_BATCHES"
echo "  Batches failed:    $FAIL_COUNT"
echo "  Genome directories: $GENOME_COUNT"
echo "  Disk usage: $(du -sh "$GENOME_DIR" | cut -f1)"
echo ""

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "Failed batches:"
    for f in "$LOG_DIR"/*.failed; do
        echo "  $(basename "$f" .failed)"
    done
    echo ""
    echo "To retry failed batches, run this script again."
fi

echo "Step 3 complete."
