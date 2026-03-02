#!/bin/bash
# Evaluate pruned IOI-mixed circuits on the VALIDATION set.
# Used to determine the best sparsity threshold for the TB3 task.
#
# Usage:
#   bash run_scripts/eval_ioi_mixed_val-set.sh
#
# Expects checkpoints at: data/runs/ioi-ioi_mixed-es{SPARSITY}-ns0.72/
# Outputs into each checkpoint's val_eval_results/ dir.
set -e

# --- Baseline: evaluate raw GPT-2 (es=0, no pruning) ---
BASELINE_DIR="./data/runs/ioi-ioi_mixed-es0-ns0.72"
BASELINE_EVAL_DIR="${BASELINE_DIR}/val_eval_results"
mkdir -p "$BASELINE_EVAL_DIR"

if [ -f "${BASELINE_EVAL_DIR}/eval_info.txt" ]; then
    echo "Skipping baseline (already evaluated)"
else
    echo "============================================"
    echo "Evaluating baseline (raw GPT-2, no pruning)"
    echo "============================================"

    python src/eval/ioi.py \
        -m gpt2 \
        -w \
        -s validation \
        -d ./data/datasets/ioi_mixed/ \
        -o "${BASELINE_EVAL_DIR}/eval_results.json" \
        > "${BASELINE_EVAL_DIR}/eval_info.txt"

    echo "Finished baseline"
fi
echo ""

# --- Pruned circuits ---
EDGE_SPARSITIES=(0.93 0.94 0.95 0.96 0.965 0.97 0.975 0.98 0.985 0.99 0.995 1.0 1.05 1.1 1.2)

for EDGE_SPARSITY in "${EDGE_SPARSITIES[@]}"; do

OUTPUT_DIR="./data/runs/ioi-ioi_mixed-es${EDGE_SPARSITY}-ns0.72"

# Skip if checkpoint doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Skipping es=$EDGE_SPARSITY (directory not found: $OUTPUT_DIR)"
    continue
fi

EVAL_DIR="${OUTPUT_DIR}/val_eval_results"
EVAL_JSON="${EVAL_DIR}/eval_results.json"
EVAL_INFO="${EVAL_DIR}/eval_info.txt"

# Skip if already evaluated
if [ -f "$EVAL_INFO" ]; then
    echo "Skipping es=$EDGE_SPARSITY (already evaluated)"
    continue
fi

mkdir -p "$EVAL_DIR"

echo "============================================"
echo "Evaluating (val): edge_sparsity=$EDGE_SPARSITY"
echo "============================================"

python src/eval/ioi.py \
    -m "$OUTPUT_DIR" \
    -w \
    -s validation \
    -d ./data/datasets/ioi_mixed/ \
    -o "$EVAL_JSON" \
    > "$EVAL_INFO"

echo "Finished es=$EDGE_SPARSITY"
echo ""

done

echo "============================================"
echo "All validation evaluations complete. Aggregating results..."
echo "============================================"

python src/eval/integrate_ioi_eval_results.py \
    --directory ./data/runs/ \
    --save_to_csv ./data/runs/ioi_mixed_val-set_eval_results.csv \
    --eval_subdir val_eval_results

python src/eval/visualize_ioi_eval_results.py \
    --eval_results_csv_path ./data/runs/ioi_mixed_val-set_eval_results.csv \
    --save_to_file ./data/runs/ioi_mixed_val-set_eval_results.png \
    --figure_title "IOI Mixed Sweep (Validation)"

echo "============================================"
echo "Done. Results:"
echo "  CSV:  ./data/runs/ioi_mixed_val-set_eval_results.csv"
echo "  Plot: ./data/runs/ioi_mixed_val-set_eval_results.png"
echo "============================================"
