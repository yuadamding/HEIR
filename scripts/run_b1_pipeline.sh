#!/usr/bin/env bash
# Full real-data image->RNA pipeline for B1_4 using spaceranger-segmented nuclei.
# Demonstrates end-to-end execution of HEIR on real H&E nuclei + matched snRNA.
# No cell/spot ground truth exists for NatCommun (donor-matched, not registered),
# so this proves feasibility + produces the morphology-graph baseline; truth-scored
# spatial validation is the locked snPATHO step.
set -uo pipefail
H=/storage/HE_GPT/HEIR
GEO=$H/artifacts/seg/B1_4/nucleus_segmentations.geojson
A=$H/artifacts/B1
mkdir -p "$A"
RUN(){ echo "### $* ###"; conda run -n hne "$@"; echo "  -> exit $?"; }

[ -f "$GEO" ] || { echo "geojson missing: $GEO"; exit 1; }

echo "===== 1. bridge geojson -> HEIR nuclei + features ====="
conda run -n hne python "$H/scripts/geojson_to_heir.py" --geojson "$GEO" \
  --out-nuclei "$A/B1_4_nuclei.csv" --out-features "$A/B1_4_features.npz" || exit 1

echo "===== 2. prepare-reference (Level1, 70-gene panel) ====="
RUN heir prepare-reference --manifest "$H/manifests/natcommun.tsv" --section-id B1_4 \
  --cell-type-key Level1 --genes "$H/manifests/gene_panel_example.tsv" \
  --gene-key feature_name --output "$A/reference.npz"

echo "===== 3. build-prototypes (fit SVD latent) ====="
RUN heir build-prototypes --reference "$A/reference.npz" \
  --reference-with-latent "$A/reference_latent.npz" \
  --fit-latent-transform "$A/shared_svd.npz" \
  --minimum-cells 30 --output "$A/prototypes.npz"

echo "===== 4. prepare-histology (real nuclei, mpp=0.5, geom-morph feature space) ====="
RUN heir prepare-histology --manifest "$H/manifests/natcommun.tsv" --section-id B1_4 \
  --nuclei "$A/B1_4_nuclei.csv" --features "$A/B1_4_features.npz" \
  --feature-space-id "geom-morph-v1" --mpp 0.5 \
  --graph-k 12 --graph-radius-um 50 --graph-max-degree 24 \
  --output "$A/histology.npz"

echo "===== 5. fit-ood ====="
RUN heir fit-ood --histology "$A/histology.npz" --quantile 0.95 --output "$A/ood.npz"

echo "===== 6. assemble-batch ====="
RUN heir assemble-batch --histology "$A/histology.npz" --prototypes "$A/prototypes.npz" \
  --reference "$A/reference.npz" --manifest "$H/manifests/natcommun.tsv" --section-id B1_4 \
  --ood-artifact "$A/ood.npz" --output "$A/batch.npz"

echo "===== 7. train (personalized; single-bag demo split) ====="
RUN heir train --train-batch "$A/batch.npz" --validation-batch "$A/batch.npz" \
  --stage personalized --epochs 40 --allow-split-overlap --allow-random-decoder \
  --device cuda --output "$A/heir_b1.pt"

echo "===== 8. predict ====="
RUN heir predict --checkpoint "$A/heir_b1.pt" --histology "$A/histology.npz" \
  --prototypes "$A/prototypes.npz" --genes "$H/manifests/gene_panel_example.tsv" \
  --donor-id 7 --sample-id B1 --ood-artifact "$A/ood.npz" \
  --device cuda --output "$A/predictions.npz"

echo "===== DONE. artifacts in $A ====="
ls -la "$A"
