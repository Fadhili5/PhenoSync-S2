#!/bin/bash
set -e

echo "Starting Track 1 inference..."

# Locate TIFF directory — try known paths in order
if [ -d "/input/region_test" ]; then
    TIFF_DIR="/input/region_test"
elif [ -d "/input/regions" ]; then
    TIFF_DIR="/input/regions"
else
    TIFF_DIR="/input"
fi
echo "Using TIFF dir: $TIFF_DIR"

python /workspace/inference.py \
    --input_csv  /input/test_point.csv \
    --tiff_dir   "$TIFF_DIR" \
    --output_dir "${OUTPUT_DIR:-/output}" \
    --model_dir  /workspace/models

echo "Done. Result at ${OUTPUT_DIR:-/output}/result.json"
