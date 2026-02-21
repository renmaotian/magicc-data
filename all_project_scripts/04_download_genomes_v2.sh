#!/bin/bash
# Step 3: Download genomes using NCBI datasets CLI (v2 - robust version)
# Uses batching (200 accessions per batch) with 10 parallel workers via xargs
# Collapse-safe: checks for existing .done markers before re-downloading

set -uo pipefail

GENOME_DIR="/home/tianrm/projects/magicc2/data/genomes"
BATCH_DIR="/home/tianrm/projects/magicc2/data/genome_batches"
LOG_DIR="/home/tianrm/projects/magicc2/data/download_logs"
NUM_WORKERS=10

mkdir -p "$GENOME_DIR" "$LOG_DIR"

NUM_BATCHES=$(ls "$BATCH_DIR"/batch_* | wc -l)
DONE_BEFORE=$(ls "$LOG_DIR"/*.done 2>/dev/null | wc -l)
echo "Total batches: $NUM_BATCHES"
echo "Already done: $DONE_BEFORE"
echo "Remaining: $((NUM_BATCHES - DONE_BEFORE))"
echo "Workers: $NUM_WORKERS"
echo "Starting at: $(date)"
echo ""

# Write the download function to a temp script so xargs can source it
cat > /tmp/download_one_batch.sh << 'SCRIPT'
#!/bin/bash
BATCH_FILE="$1"
BATCH_NAME=$(basename "$BATCH_FILE")
GENOME_DIR="/home/tianrm/projects/magicc2/data/genomes"
LOG_DIR="/home/tianrm/projects/magicc2/data/download_logs"
BATCH_LOG="$LOG_DIR/${BATCH_NAME}.log"
DONE_MARKER="$LOG_DIR/${BATCH_NAME}.done"

# Skip if already done
if [ -f "$DONE_MARKER" ]; then
    exit 0
fi

TMPZIP="/tmp/ncbi_dl_${BATCH_NAME}_$$.zip"
TMPEXTRACT="/tmp/ncbi_extract_${BATCH_NAME}_$$"

MAX_RETRIES=3
RETRY=0
SUCCESS=0

while [ $RETRY -lt $MAX_RETRIES ] && [ $SUCCESS -eq 0 ]; do
    RETRY=$((RETRY + 1))
    rm -f "$TMPZIP"
    rm -rf "$TMPEXTRACT"

    if datasets download genome accession \
        --inputfile "$BATCH_FILE" \
        --include genome \
        --filename "$TMPZIP" \
        --no-progressbar >> "$BATCH_LOG" 2>&1; then

        if [ -f "$TMPZIP" ] && unzip -t "$TMPZIP" > /dev/null 2>&1; then
            mkdir -p "$TMPEXTRACT"
            if unzip -q -o "$TMPZIP" -d "$TMPEXTRACT" >> "$BATCH_LOG" 2>&1; then
                MOVED=0
                if [ -d "$TMPEXTRACT/ncbi_dataset/data" ]; then
                    for acc_dir in "$TMPEXTRACT"/ncbi_dataset/data/GC*; do
                        if [ -d "$acc_dir" ]; then
                            acc=$(basename "$acc_dir")
                            target_dir="$GENOME_DIR/$acc"
                            if [ ! -d "$target_dir" ]; then
                                mv "$acc_dir" "$target_dir" 2>/dev/null && MOVED=$((MOVED + 1)) || true
                            else
                                MOVED=$((MOVED + 1))
                            fi
                        fi
                    done
                fi
                SUCCESS=1
                echo "[DONE] $BATCH_NAME: $MOVED genomes (attempt $RETRY)"
            fi
        else
            echo "[RETRY] $BATCH_NAME: invalid zip (attempt $RETRY)" >> "$BATCH_LOG"
            sleep $((RETRY * 10))
        fi
    else
        echo "[RETRY] $BATCH_NAME: download failed (attempt $RETRY)" >> "$BATCH_LOG"
        sleep $((RETRY * 10))
    fi
done

rm -f "$TMPZIP"
rm -rf "$TMPEXTRACT"

if [ $SUCCESS -eq 1 ]; then
    touch "$DONE_MARKER"
else
    touch "$LOG_DIR/${BATCH_NAME}.failed"
    echo "[FAIL] $BATCH_NAME after $MAX_RETRIES retries"
fi
SCRIPT
chmod +x /tmp/download_one_batch.sh

# Run all batches with parallel workers
ls "$BATCH_DIR"/batch_* | xargs -P $NUM_WORKERS -I {} /tmp/download_one_batch.sh {}

# Report results
DONE_COUNT=$(ls "$LOG_DIR"/*.done 2>/dev/null | wc -l)
FAIL_COUNT=$(ls "$LOG_DIR"/*.failed 2>/dev/null | wc -l)
GENOME_COUNT=$(find "$GENOME_DIR" -maxdepth 1 -type d -name "GC*" | wc -l)

echo ""
echo "========================================="
echo "Download Summary ($(date))"
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
fi

echo "Step 3 complete."
