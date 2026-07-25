# run_pipeline.ps1
# Master script to run evaluation pipeline on specified models and datasets.

$MODELS = @(
    "Codestral",
    "Cohere-Command-AISec",
    "Cohere-command-AP",
    "DeepSeek-V3.2-AISec",
    "DeepSeek-V4-Pro",
    "gpt-5-nano",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "grok-4-20-nr",
    "grok-4-20-reasoning",
    "grok-4.3",
    "Kimi-K2.6",
    "Kimi-K2.7-Code",
    "Llama-3.3-70B-Instruct",
    "Llama-4-Maverick-17B-128E-Instruct-FP8",
    "Mistral-Large-3"
)

$DATASETS = @(
    "devsecops_prompts.csv",
    "devsecops_advanced_prompts.csv"
)

Write-Host "Starting DevSecOps Evaluation Pipeline"
Write-Host "======================================"

foreach ($MODEL in $MODELS) {
    foreach ($DATASET in $DATASETS) {
        Write-Host ""
        Write-Host ">>> Evaluating Model: $MODEL on Dataset: $DATASET <<<"
        
        # Determine paths
        $DATASET_NAME = [System.IO.Path]::GetFileNameWithoutExtension($DATASET)
        $MODEL_DIR = "results/${MODEL}"
        $RAW_OUTPUT = "${MODEL_DIR}/${MODEL}_${DATASET_NAME}_results.json"
        $SCORED_OUTPUT = "${MODEL_DIR}/scored_responses_${MODEL}_${DATASET_NAME}.json"
        $SCORED_CSV = "${MODEL_DIR}/scored_responses_${MODEL}_${DATASET_NAME}.csv"
        
        # 1. Generate LLM Responses
        Write-Host "[1/4] Generating responses..."
        python scripts/generate_llm_responses_json.py --model "$MODEL" --dataset "$DATASET"
        
        # 2. Run LLM Judge
        Write-Host "[2/4] Running LLM Judge..."
        if (Test-Path -Path $RAW_OUTPUT -PathType Leaf) {
            python scripts/llm_judge.py --input "$RAW_OUTPUT" --output "$SCORED_OUTPUT"
        }
        else {
            Write-Host "Error: Raw output file $RAW_OUTPUT not found! Skipping..."
            continue
        }
        
        # 3. Print Summary
        Write-Host "[3/4] Generating Summary..."
        if (Test-Path -Path $SCORED_OUTPUT -PathType Leaf) {
            $SUMMARY_OUTPUT = "${MODEL_DIR}/summary_${MODEL}_${DATASET_NAME}.txt"
            python scripts/summary.py --input "$SCORED_OUTPUT" | Tee-Object -FilePath $SUMMARY_OUTPUT
        }
        else {
            Write-Host "Skipping summary because $SCORED_OUTPUT does not exist."
        }
        
        Write-Host ">>> Finished evaluation for $MODEL on $DATASET <<<"
    }
}

Write-Host "======================================"
Write-Host ">>> Generating Final Aggregated Plots across all models <<<"
python scripts/generate_plots.py
Write-Host "Pipeline execution complete!"
