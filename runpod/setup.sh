#!/bin/bash
# RunPod setup script — run once after spinning up the pod
# Usage: bash runpod/setup.sh
set -euo pipefail

echo "=== RunPod Setup ==="

# Install project dependencies
pip install -q openai>=1.0 transformers>=4.40 torch>=2.0 scikit-learn>=1.4 tqdm accelerate

# The test dataset is committed to the repo (data/ebm_nlp_test_masked.json).
# Only run data_prep.py if it's missing.
if [[ -f "data/ebm_nlp_test_masked.json" ]]; then
    echo "=== Dataset already present, skipping data_prep.py ==="
else
    echo "=== Preparing dataset ==="
    python src/data_prep.py
fi

echo "=== Verifying data ==="
python -c "
import json
with open('data/ebm_nlp_test_masked.json') as f:
    data = json.load(f)
print(f'test: {len(data)} examples')
"

echo "=== Setup complete ==="
echo "Run experiments with: bash runpod/run_all.sh"
