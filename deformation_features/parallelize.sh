#!/bin/bash

bundles=("AF_L" "PYT_L" "ILF_L" "FAT_L" "IFOF_L")
n_parallel=120

PROJECT_DIR="/home/thuythienthanh.tran/mnt/pc"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

cd "$PROJECT_DIR" || exit 1
OUTPUT_DIR="$PROJECT_DIR/output"
LOG_DIR="$PROJECT_DIR/output/logs"

mkdir -p "$LOG_DIR"

triples=$(python -c "
import json
import re

output_dir = '$OUTPUT_DIR'

bundles = ['AF_L', 'PYT_L', 'ILF_L', 'FAT_L', 'IFOF_L']
pattern = re.compile(r'sub-(\d+)_tum-(\d+)')

for bundle in bundles:
    with open(f'{output_dir}/bundle_idx_{bundle}.json') as f:
        bundle_idx = json.load(f)

    paths = [e['path'] for e in bundle_idx[bundle]]

    for p in paths:
        match = pattern.search(p)
        if match:
            sub, tum = match.groups()
            print(bundle, sub, tum)
")

while IFS=' ' read -r bundle sub tum; do

    log_file="${LOG_DIR}/extract_${bundle}_sub-${sub}_tum-${tum}.txt"

    python -u training_script/deformation_features/deformation_features.py \
        --bundle "$bundle" \
        --sub "$sub" \
        --tum "$tum" \
        &> "$log_file" &

    if (( $(jobs -r -p | wc -l) >= n_parallel )); then
        wait -n
    fi

done <<< "$triples"

wait

echo "All bundles done."
