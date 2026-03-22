#!/bin/bash
# RunPod setup script — run once after spinning up the pod
# Usage: bash runpod/setup.sh
set -euo pipefail

echo "=== RunPod Setup ==="

# Install project dependencies
pip install -q openai>=1.0 transformers>=4.40 torch>=2.0 scikit-learn>=1.4 tqdm accelerate

# Download the EBM-NLP dataset and prepare masked data
echo "=== Preparing dataset ==="
python src/data_prep.py

echo "=== Verifying data ==="
python -c "
import json
for split in ['train', 'test']:
    with open(f'data/ebm_nlp_{split}_masked.json') as f:
        data = json.load(f)
    print(f'{split}: {len(data)} examples')
"

echo "=== Setup complete ==="
echo "Run experiments with: bash runpod/run_all.sh"
