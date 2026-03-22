#!/bin/bash
# Run all GPU-dependent experiments on RunPod.
# Usage: bash runpod/run_all.sh [--skip-70b]
#
# Results are saved to results/ — copy them back after the run.
# Estimated time on A100 80GB: ~1 hour total.
set -euo pipefail

SKIP_70B=false
if [[ "${1:-}" == "--skip-70b" ]]; then
    SKIP_70B=true
    echo "Skipping 70B model runs."
fi

echo "=== GPU Info ==="
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB' if torch.cuda.is_available() else 'No GPU')"

run_model() {
    local model=$1
    local label=$2

    echo ""
    echo "=========================================="
    echo "  Running: $label"
    echo "=========================================="

    echo "--- Zero-shot ---"
    python src/run_experiment.py --task 1 --model "$model"

    echo "--- Few-shot ---"
    python src/run_experiment.py --task 1 --model "$model" --few-shot
}

# --- Priority 1: Models from the original proposal ---

run_model "BioMistral/BioMistral-7B" "BioMistral-7B (domain-specialized base)"
run_model "meta-llama/Llama-3.1-8B-Instruct" "Llama-3.1-8B-Instruct (proposed model)"

# --- Priority 2: Controlled comparison ---

run_model "mistralai/Mistral-7B-Instruct-v0.3" "Mistral-7B-Instruct-v0.3 (BioMistral parent, instruct)"

# --- Priority 3: Scale test (optional) ---

if [[ "$SKIP_70B" == false ]]; then
    run_model "meta-llama/Llama-3.1-70B-Instruct" "Llama-3.1-70B-Instruct (scale test)"
else
    echo ""
    echo "Skipped Llama-3.1-70B-Instruct (--skip-70b flag)."
fi

echo ""
echo "=========================================="
echo "  All runs complete!"
echo "=========================================="
echo ""
echo "Results saved in results/:"
ls -lh results/task1_*.json
echo ""
echo "To evaluate locally:"
echo "  python src/evaluate.py --task 1 --results results/<file>.json"
echo ""
echo "Copy results back:"
echo "  scp -r results/ your-machine:~/llm-proj/results/"
