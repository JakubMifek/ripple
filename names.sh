#!/usr/bin/env bash
set -euo pipefail

# New output directory required; src/compare.py never overwrites committed examples.
output_dir="${1:?Usage: bash names.sh NEW_OUTPUT_DIR [OPTIONS]}"
shift
python3 src/compare.py samples/Baltazar.wav samples/Leontynka.wav samples/Eliska.wav samples/Kuba.wav \
    --output-dir "$output_dir" \
    --smoothness 0.309 --color black "$@"
