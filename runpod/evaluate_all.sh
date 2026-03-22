#!/bin/bash
# Evaluate all Task 1 result files in results/
# Usage: bash runpod/evaluate_all.sh
set -euo pipefail

echo "=== Evaluating all Task 1 results ==="
echo ""

for f in results/task1_*.json; do
    # Skip the failed BioMistral local run (5 errors)
    if [[ "$f" == *"BioMistral"* ]] && python -c "
import json
with open('$f') as fh:
    d = json.load(fh)
r = d['results'] if isinstance(d, dict) else d
if len(r) < 10: exit(0)
exit(1)
" 2>/dev/null; then
        echo "Skipping $f (incomplete local run)"
        echo ""
        continue
    fi

    echo "--- $f ---"
    python src/evaluate.py --task 1 --results "$f"
    echo ""
done
