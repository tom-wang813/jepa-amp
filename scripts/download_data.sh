#!/usr/bin/env bash
# Download AMP datasets into data/raw/
set -e
DATA_DIR="$(dirname "$0")/../data/raw"
mkdir -p "$DATA_DIR"

echo "=== Downloading APD3 (Antimicrobial Peptide Database) ==="
# APD3 provides a FASTA export of all peptides
APD3_URL="https://aps.unmc.edu/assets/sequences/APD_sequence_release_09142020.fasta"
wget -q --show-progress -O "$DATA_DIR/apd3.fasta" "$APD3_URL" || {
    echo "APD3 direct download failed. Please manually download from https://aps.unmc.edu"
    echo "Save as: $DATA_DIR/apd3.fasta"
}

echo ""
echo "=== Downloading DBAASP subset (via public mirror) ==="
# DBAASP full export requires registration; use the curated subset from prior works
DBAASP_URL="https://raw.githubusercontent.com/AliYoussef96/BCPNN-AMP/master/data/AMPs.fasta"
wget -q --show-progress -O "$DATA_DIR/dbaasp_amps.fasta" "$DBAASP_URL" || {
    echo "DBAASP mirror failed."
}

echo ""
echo "Done. Files saved to $DATA_DIR/"
ls -lh "$DATA_DIR/"
