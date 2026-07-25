import json
import csv
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Convert LLM JSON results to CSV for grading.")
    parser.add_argument("--input", type=str, required=True, help="Path to the JSON results file.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    model_name = data.get("metadata", {}).get("model", "unknown-model")
    
    if not results:
        print("No results found in the JSON file.")
        return

    # Prepare CSV output path
    output_path = input_path.parent / f"raw_responses_{model_name}.csv"

    # We need to map JSON fields to the CSV headers expected by llm_judge.py and generate_plots.py
    # llm_judge expects: prompt_id, prompt_text, response_text, model_name, (error, category, etc.)
    fieldnames = [
        "prompt_id",
        "category",
        "difficulty_level",
        "difficulty_label",
        "prompt_text",
        "expected_artifact",
        "model_name",
        "response_text",
        "prompt_tokens",
        "response_tokens",
        "total_tokens",
        "error"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for res in results:
            token_metrics = res.get("token_metrics", {})
            writer.writerow({
                "prompt_id": res.get("prompt_id", ""),
                "category": res.get("category", ""),
                "difficulty_level": res.get("difficulty_level", ""),
                "difficulty_label": res.get("difficulty_label", ""),
                "prompt_text": res.get("prompt_text", ""),
                "expected_artifact": res.get("expected_artifact", ""),
                "model_name": model_name,
                "response_text": res.get("response", ""),
                "prompt_tokens": token_metrics.get("prompt_tokens", 0),
                "response_tokens": token_metrics.get("response_tokens", 0),
                "total_tokens": token_metrics.get("total_tokens", 0),
                "error": res.get("error", "")
            })

    print(f"Converted {len(results)} records.")
    print(f"Saved CSV to: {output_path}")
    print(f"You can now run: python scripts/llm_judge.py --input {output_path}")

if __name__ == "__main__":
    main()
