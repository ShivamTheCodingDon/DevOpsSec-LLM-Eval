#!/bin/bash

# run_pipeline.sh
# Master script to run evaluation pipeline on specified models and datasets.

# List of models to evaluate.
# List of models to evaluate.
# You can add other models to this list
MODELS="Codestral Mistral-Large-3 gpt-5-nano gpt-5.4 gpt-5.4-mini gpt-5.4-nano grok-4-20-reasoning Kimi-K2.6 grok-4.3 DeepSeek-V4-Pro"

# List of datasets to evaluate against.

DATASETS="devsecops_prompts.csv devsecops_advanced_prompts.csv"

echo "Starting DevSecOps Evaluation Pipeline"
echo "======================================"

for MODEL in $MODELS; do
    for DATASET in $DATASETS; do
        echo ""
        echo ">>> Evaluating Model: $MODEL on Dataset: $DATASET <<<"
        
        # 1. Generate LLM Responses
        echo "[1/4] Generating responses..."
        python -u scripts/generate_llm_responses_json.py --model "$MODEL" --dataset "$DATASET"
        
        # Determine the base dataset name without extension
        DATASET_NAME="${DATASET%.*}"
        MODEL_DIR="results/${MODEL}"
        RAW_OUTPUT="${MODEL_DIR}/${MODEL}_${DATASET_NAME}_results.json"
        SCORED_OUTPUT="${MODEL_DIR}/scored_responses_${MODEL}_${DATASET_NAME}.json"
        SCORED_CSV="${MODEL_DIR}/scored_responses_${MODEL}_${DATASET_NAME}.csv"
        
        # 2. Run LLM Judge
        echo "[2/4] Running LLM Judge..."
        if [ -f "$RAW_OUTPUT" ]; then
            python -u scripts/llm_judge.py --input "$RAW_OUTPUT" --output "$SCORED_OUTPUT"
        else
            echo "Error: Raw output file $RAW_OUTPUT not found! Skipping..."
            continue
        fi
        
        # 3. Print Summary
        echo "[3/4] Generating Summary..."
        if [ -f "$SCORED_OUTPUT" ]; then
            python -u scripts/summary.py --input "$SCORED_OUTPUT"
        else
            echo "Skipping summary because $SCORED_OUTPUT does not exist."
        fi
        
        # 4. Generate Plots
        echo "[4/4] Generating Plots..."
        if [ -f "$SCORED_CSV" ]; then
            python -u scripts/generate_plots.py --input "$SCORED_CSV"
        else
            echo "Warning: Scored CSV file $SCORED_CSV not found. Plots might not be generated if not applicable."
        fi
        
        echo ">>> Finished evaluation for $MODEL on $DATASET <<<"
    done
done

echo ""
echo "Pipeline completed successfully!"
