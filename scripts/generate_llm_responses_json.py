import os
import csv
import json
import time
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from openai import AzureOpenAI

import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate LLM responses for a dataset")
    parser.add_argument("--model", type=str, default="Kimi-K2.7-Code", help="Deployment name of the model")
    parser.add_argument("--dataset", type=str, default="devsecops_advanced_prompts.csv", help="Dataset file name in the dataset folder")
    args = parser.parse_args()

    DEPLOYMENT_NAME = args.model
    DATASET_NAME = Path(args.dataset).stem
    DATASET_PATH = Path(__file__).parent.parent / "dataset" / args.dataset
    RESULTS_DIR = Path(__file__).parent.parent / "results" / DEPLOYMENT_NAME
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE = RESULTS_DIR / f"{DEPLOYMENT_NAME}_{DATASET_NAME}_results.json"
    
    # Load .env file
    load_dotenv(Path(__file__).parent.parent / ".env")
    
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    
    IS_GEMINI = "gemini" in DEPLOYMENT_NAME.lower()
    
    if IS_GEMINI:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            print("Error: GEMINI_API_KEY not found in .env")
            return
        from google import genai
        client = genai.Client(api_key=gemini_api_key)
    else:
        if not api_key or not endpoint:
            print("Error: Azure OpenAI credentials not found in .env")
            return
    
        # Initialize the LLM client
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )

    # Directory is already created above
    
    completed_prompts = set()
    results = []
    category_tokens = defaultdict(int)
    total_tokens_all = 0
    total_response_tokens_all = 0
    
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                results = data.get("results", [])
                completed_prompts = {r.get("prompt_id") for r in results if r.get("prompt_id")}
                # Restore metrics
                metadata = data.get("metadata", {})
                total_response_tokens_all = metadata.get("total_response_tokens", 0)
                total_tokens_all = metadata.get("total_tokens_overall", 0)
                cat_totals = metadata.get("category_totals", {})
                for k, v in cat_totals.items():
                    category_tokens[k] = v
            print(f"Resuming from existing file. {len(completed_prompts)} prompts already processed.", flush=True)
        except Exception as e:
            print(f"Error loading existing results: {e}. Starting fresh.", flush=True)
    
    print(f"Starting evaluation for model: {DEPLOYMENT_NAME}", flush=True)
    
    # Read the dataset
    all_rows = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["prompt_id"].strip().startswith("#"):
                continue
            all_rows.append(row)
            
    total_prompts = len(all_rows)
    print(f"Found {total_prompts} prompts to process.", flush=True)
    
    for i, row in enumerate(all_rows, 1):
        prompt_id = row["prompt_id"]
        
        if prompt_id in completed_prompts:
            print(f"Skipping [{i}/{total_prompts}] {prompt_id} - Already processed", flush=True)
            continue
            
        prompt_text = row["prompt_text"]
        category = row["category"]
            
        try:
            if IS_GEMINI:
                generation_config = {
                    'max_output_tokens': 2048,
                    'temperature': 0.0,
                    'top_p': 0.95,
                    'thinking_level': 'medium',
                }
                
                max_retries = 4
                retry_delays = [10, 20, 30]
                
                for attempt in range(max_retries):
                    try:
                        interaction = client.interactions.create(
                            model=DEPLOYMENT_NAME,
                            input=prompt_text,
                            generation_config=generation_config,
                        )
                        break
                    except Exception as e:
                        if attempt < len(retry_delays):
                            print(f"Rate limit or error hit for {prompt_id}. Waiting {retry_delays[attempt]}s. Error: {e}", flush=True)
                            time.sleep(retry_delays[attempt])
                        else:
                            raise e

                content = interaction.output_text
                finish_reason = "unknown"
                prompt_tokens = 0
                response_tokens = 0
                total_tokens = 0
                
                if content is None:
                    content = ""
                if not content.strip():
                    content = f"I cannot fulfill this request. [Content Filtered or Empty Response from API] (finish_reason: {finish_reason})"
            else:
                # Call the LLM
                kwargs = {
                    "model": DEPLOYMENT_NAME,
                    "messages": [{"role": "user", "content": prompt_text}],
                }
                if "gpt" in DEPLOYMENT_NAME.lower():
                    kwargs["max_completion_tokens"] = 2048
                else:
                    kwargs["max_tokens"] = 2048
                    kwargs["temperature"] = 0.0
                    
                if "phi" in DEPLOYMENT_NAME.lower():
                    kwargs["timeout"] = 360  # increased timeout for phi models
                else:
                    kwargs["timeout"] = 60
                    
                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason if hasattr(response.choices[0], 'finish_reason') else "unknown"
                
                if content is None:
                    content = ""
                    
                if not content.strip():
                    content = f"I cannot fulfill this request. [Content Filtered or Empty Response from API] (finish_reason: {finish_reason})"
                    prompt_tokens = 0
                    response_tokens = 0
                    total_tokens = 0
                else:
                    # Extract token usage if available
                    if response.usage:
                        prompt_tokens = response.usage.prompt_tokens
                        response_tokens = response.usage.completion_tokens
                        total_tokens = response.usage.total_tokens
                    else:
                        prompt_tokens = 0
                        response_tokens = 0
                        total_tokens = 0
            
            # Update metrics
            category_tokens[category] += total_tokens
            total_tokens_all += total_tokens
            total_response_tokens_all += response_tokens
            
            # Build detailed JSON entry
            result_entry = {
                "prompt_id": prompt_id,
                "category": category,
                "difficulty_level": row.get("difficulty_level", ""),
                "difficulty_label": row.get("difficulty_label", ""),
                "prompt_text": prompt_text,
                "expected_artifact": row.get("expected_artifact", ""),
                "response": content,
                "token_metrics": {
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                    "total_tokens": total_tokens
                }
            }
            results.append(result_entry)
            print(f"Processed [{i}/{total_prompts}] {prompt_id} - Response tokens: {response_tokens}", flush=True)
            
        except Exception as e:
            err_msg = str(e)
            print(f"Error on {prompt_id}: {err_msg}", flush=True)
            
            # If the API blocked it, record that as the response
            fallback_content = f"API Error / Content Filtered: {err_msg}"
            
            # Record it as a result so we still get a row in the CSV later
            result_entry = {
                "prompt_id": prompt_id,
                "category": category,
                "difficulty_level": row.get("difficulty_level", ""),
                "difficulty_label": row.get("difficulty_label", ""),
                "prompt_text": prompt_text,
                "expected_artifact": row.get("expected_artifact", ""),
                "response": fallback_content,
                "token_metrics": {
                    "prompt_tokens": 0,
                    "response_tokens": 0,
                    "total_tokens": 0
                },
                "error": err_msg
            }
            results.append(result_entry)
            
        # Save intermediate progress to avoid losing data if interrupted
        output_data = {
            "metadata": {
                "model": DEPLOYMENT_NAME,
                "total_response_tokens": total_response_tokens_all,
                "total_tokens_overall": total_tokens_all,
                "category_totals": dict(category_tokens)
            },
            "results": results
        }
        with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
            json.dump(output_data, out_f, indent=4, ensure_ascii=False)

                
            # Rate limiting pause
            time.sleep(1)
            
    # Final save of all results
    output_data = {
        "metadata": {
            "model": DEPLOYMENT_NAME,
            "total_response_tokens": total_response_tokens_all,
            "total_tokens_overall": total_tokens_all,
            "category_totals": dict(category_tokens)
        },
        "results": results
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        json.dump(output_data, out_f, indent=4, ensure_ascii=False)
        
    print(f"\nEvaluation Complete! Results saved to {OUTPUT_FILE}")
    print(f"Overall Total Tokens: {total_tokens_all}")
    print(f"Overall Response Tokens: {total_response_tokens_all}")
    print("Category Totals:")
    for cat, toks in category_tokens.items():
        print(f"  {cat}: {toks}")

if __name__ == "__main__":
    main()
