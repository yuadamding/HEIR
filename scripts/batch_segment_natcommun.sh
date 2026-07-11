#!/usr/bin/env bash
# Segment all INCLUDED NatCommun H&E sections (skips B2) with spaceranger segment.
# Resumable: skips sections whose geojson already exists. Sequential (single GPU).
# Usage: bash batch_segment_natcommun.sh
set -uo pipefail
H=/storage/HE_GPT/HEIR
PROC=/mnt/seagate/HnE/NatCommun_2025_s41467_025_59005_9/arrayexpress/E-MTAB-14560/processed_data
# section_id -> H&E filename (from manifests/natcommun.tsv, included rows only)
declare -A HE=(
  [B1_2]=B1_2.tif [B1_4]=B1_4.tif [B3_2]=B3_2.tif [B4_2]=B4_2.tif
  [L1_2]=L1_2.tif [L1_4]=L1_4.tif [L2_2]=L2_2.tif [L3_2]=L3_2.tif [L4_2]=L4_2.tif
  [D1]=D1.tif [D2]=D2.tif [D3]=D3.tif [D4]=D4.tif [D5]=D5.tif [D6]=D6.tif
)
for sec in "${!HE[@]}"; do
  out=$H/artifacts/seg/$sec
  if [ -f "$out/nucleus_segmentations.geojson" ]; then echo "[skip] $sec"; continue; fi
  echo "==== segmenting $sec ===="
  bash "$H/scripts/segment_slide.sh" "$PROC/${HE[$sec]}" "$sec" "$out"
done
echo "=== segmentation batch complete ==="
for sec in "${!HE[@]}"; do
  g=$H/artifacts/seg/$sec/nucleus_segmentations.geojson
  n=$( [ -f "$g" ] && grep -o '"Polygon"' "$g" | wc -l || echo 0 )
  echo "$sec: $n nuclei"
done
