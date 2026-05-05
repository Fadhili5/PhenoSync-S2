#!/bin/bash
set -e

echo "Starting Track 1 inference..."
python /workspace/inference.py \
    --input_csv  /input/points_test.csv \
    --tiff_dir   /input/region_test \
    --output_dir ${OUTPUT_DIR:-/output} \
    --model_dir  /workspace/models

echo "Done. Result at ${OUTPUT_DIR:-/output}/result.json"
