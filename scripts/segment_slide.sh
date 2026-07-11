#!/usr/bin/env bash
# Convert one NatCommun H&E TIFF -> tiled pyramidal BigTIFF (pyvips, r_env) ->
# spaceranger segment (GPU StarDist) -> collect nucleus_segmentations.geojson.
# Mirrors the proven AVPC recipe. Args: <he_tif> <run_id> <out_dir>
set -uo pipefail
SR=/storage/hackathon_2026/tools/spaceranger-4.1.0/spaceranger
HE="$1"; RUNID="$2"; OUTDIR="$3"
WORK=/storage/HE_GPT/HEIR/artifacts/seg_work
LOGDIR="$OUTDIR/logs"
mkdir -p "$OUTDIR" "$WORK" "$LOGDIR"
GEOJSON="$OUTDIR/nucleus_segmentations.geojson"
if [ -f "$GEOJSON" ]; then echo "[skip] $RUNID (geojson exists)"; exit 0; fi
TIF="$WORK/${RUNID}.tif"
RUNDIR="$WORK/$RUNID"
echo "[start] $RUNID $(date '+%H:%M:%S')  src=$HE"
# 1) -> tiled pyramidal BigTIFF, drop alpha if present
conda run -n r_env python -c '
import sys, pyvips
src, tif = sys.argv[1], sys.argv[2]
im = pyvips.Image.new_from_file(src, access="sequential")
if im.bands == 4: im = im.extract_band(0, n=3)
if im.bands == 1: im = im.colourspace("srgb")
im.tiffsave(tif, tile=True, pyramid=True, compression="jpeg", Q=90, bigtiff=True)
' "$HE" "$TIF" > "$LOGDIR/${RUNID}.conv.log" 2>&1 || { echo "[FAIL conv] $RUNID"; exit 1; }
[ -f "$TIF" ] || { echo "[FAIL conv-notif] $RUNID"; exit 1; }
# 2) spaceranger segment
rm -rf "$RUNDIR"
( cd "$WORK" && "$SR" segment --id "$RUNID" --tissue-image "$TIF" \
      --localcores 8 --localmem 24 --disable-ui ) > "$LOGDIR/${RUNID}.sr.log" 2>&1 \
  || { echo "[FAIL sr] $RUNID"; rm -f "$TIF"; exit 1; }
# 3) collect + cleanup
cp -f "$RUNDIR/outs/nucleus_segmentations.geojson" "$OUTDIR/" 2>/dev/null
cp -f "$RUNDIR/outs/nucleus_instance_mask.tiff" "$OUTDIR/" 2>/dev/null
cp -f "$RUNDIR/outs/web_summary.html" "$OUTDIR/" 2>/dev/null
rm -f "$TIF"; rm -rf "$RUNDIR"
NC=$(grep -o '"Polygon"' "$GEOJSON" 2>/dev/null | wc -l)
echo "[done] $RUNID nuclei=$NC $(date '+%H:%M:%S')"
