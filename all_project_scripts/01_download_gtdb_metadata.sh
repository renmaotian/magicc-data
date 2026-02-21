#!/bin/bash
# Step 1: Download GTDB metadata files (bacterial and archaeal)
# Saves to /home/tianrm/projects/magicc2/data/gtdb/

set -euo pipefail

OUTDIR="/home/tianrm/projects/magicc2/data/gtdb"
mkdir -p "$OUTDIR"

BAC_URL="https://data.gtdb.aau.ecogenomic.org/releases/latest/bac120_metadata.tsv.gz"
AR_URL="https://data.gtdb.aau.ecogenomic.org/releases/latest/ar53_metadata.tsv.gz"

# Download bacterial metadata (skip if already exists and non-empty)
BAC_FILE="$OUTDIR/bac120_metadata.tsv.gz"
if [ -s "$BAC_FILE" ]; then
    echo "Bacterial metadata already exists: $BAC_FILE ($(du -h "$BAC_FILE" | cut -f1))"
else
    echo "Downloading bacterial metadata..."
    wget -c -O "$BAC_FILE" "$BAC_URL"
    echo "Downloaded: $(du -h "$BAC_FILE" | cut -f1)"
fi

# Download archaeal metadata (skip if already exists and non-empty)
AR_FILE="$OUTDIR/ar53_metadata.tsv.gz"
if [ -s "$AR_FILE" ]; then
    echo "Archaeal metadata already exists: $AR_FILE ($(du -h "$AR_FILE" | cut -f1))"
else
    echo "Downloading archaeal metadata..."
    wget -c -O "$AR_FILE" "$AR_URL"
    echo "Downloaded: $(du -h "$AR_FILE" | cut -f1)"
fi

# Verify downloads
echo ""
echo "=== Download Summary ==="
for f in "$BAC_FILE" "$AR_FILE"; do
    if [ -s "$f" ]; then
        echo "OK: $f ($(du -h "$f" | cut -f1))"
    else
        echo "FAILED: $f"
        exit 1
    fi
done

echo ""
echo "Step 1 complete: GTDB metadata files downloaded successfully."
