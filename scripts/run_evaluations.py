"""
DevSecOps LLM Safety Evaluation - API Query Pipeline
=====================================================
Reads prompts from the CSV dataset, sends them to multiple LLM APIs
(OpenAI, Anthropic, Google, DeepSeek, Qwen/DashScope), and saves all
responses to a results CSV.

Usage:
    pip install openai anthropic google-generativeai python-dotenv tqdm
    python run_evaluations.py

Set your API keys in a .env file:
    OPENAI_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-...
    GOOGLE_API_KEY=AI...
    DEEPSEEK_API_KEY=sk-...
    DASHSCOPE_API_KEY=sk-...
"""

import csv
import json
import os
import time
import asyncio
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

DATASET_PATH = Path(__file__).parent.parent / "dataset" / "devsecops_prompts.csv"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_FILE = RESULTS_DIR / f"raw_responses_{TIMESTAMP}.csv"

# Models to evaluate (5 frontier models for empirical comparison)
MODELS = {
    "gpt-4o": {
        "provider": "openai",
        "model_id": "gpt-4o",
    },
    "claude-3.5-sonnet": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-20250514",
    },
    "gemini-1.5-pro": {
        "provider": "google",
        "model_id": "gemini-1.5-pro",
    },
    "deepseek-v3": {
        "provider": "deepseek",
        "model_id": "deepseek-chat",
    },
    "qwen-2.5-72b": {
        "provider": "dashscope",
        "model_id": "qwen-plus",
    },
}

# Rate limiting (seconds between requests per provider)
RATE_LIMIT_DELAY = {
    "openai": 1.0,
    "anthropic": 1.2,
    "google": 1.0,
    "deepseek": 1.0,
    "dashscope": 1.0,
}

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


# ---------------------------------------------------------------------------
# Provider query functions
# ---------------------------------------------------------------------------
def query_openai(prompt: str, model_id: str) -> str:
    """Query OpenAI API."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {str(e)}"


def query_anthropic(prompt: str, model_id: str) -> str:
    """Query Anthropic API."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    try:
        response = client.messages.create(
            model=model_id,
            max_tokens=2048,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {str(e)}"


def query_google(prompt: str, model_id: str) -> str:
    """Query Google Generative AI API."""
    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    try:
        model = genai.GenerativeModel(model_id)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=2048,
                temperature=0.0,
            ),
        )
        return response.text
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {str(e)}"


def query_deepseek(prompt: str, model_id: str) -> str:
    """Query DeepSeek API (OpenAI-compatible endpoint)."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {str(e)}"


def query_dashscope(prompt: str, model_id: str) -> str:
    """Query Alibaba DashScope API for Qwen models (OpenAI-compatible)."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {str(e)}"


PROVIDER_FUNCTIONS = {
    "openai": query_openai,
    "anthropic": query_anthropic,
    "google": query_google,
    "deepseek": query_deepseek,
    "dashscope": query_dashscope,
}


# ---------------------------------------------------------------------------
# Load prompts
# ---------------------------------------------------------------------------
def load_prompts(csv_path: Path) -> list[dict]:
    """Load prompts from the CSV dataset, skipping comments."""
    prompts = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip comment rows (prompt_id starts with #)
            if row["prompt_id"].strip().startswith("#"):
                continue
            prompts.append(row)
    print(f"Loaded {len(prompts)} prompts from {csv_path}")
    return prompts


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
def query_with_retry(
    provider_fn, prompt_text: str, model_id: str, retries: int = MAX_RETRIES
) -> str:
    """Query a model with retry logic."""
    for attempt in range(retries):
        result = provider_fn(prompt_text, model_id)
        if not result.startswith("[ERROR]"):
            return result
        if attempt < retries - 1:
            print(f"  Retry {attempt + 1}/{retries} after error: {result}")
            time.sleep(RETRY_DELAY * (attempt + 1))
    return result  # Return the last error


def run_evaluation():
    """Main evaluation loop."""
    prompts = load_prompts(DATASET_PATH)

    # Prepare results CSV
    fieldnames = [
        "prompt_id",
        "category",
        "difficulty_level",
        "difficulty_label",
        "prompt_text",
        "expected_artifact",
        "model_name",
        "provider",
        "response_text",
        "response_length",
        "timestamp",
        "error",
    ]

    # Check for existing progress to support resumption
    completed = set()
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add((row["prompt_id"], row["model_name"]))
        print(f"Resuming: {len(completed)} responses already collected.")

    mode = "a" if RESULTS_FILE.exists() else "w"
    with open(RESULTS_FILE, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()

        total_tasks = len(prompts) * len(MODELS)
        skipped = 0

        with tqdm(total=total_tasks, desc="Evaluating") as pbar:
            for prompt in prompts:
                for model_name, model_config in MODELS.items():
                    # Skip if already completed
                    if (prompt["prompt_id"], model_name) in completed:
                        pbar.update(1)
                        skipped += 1
                        continue

                    provider = model_config["provider"]
                    model_id = model_config["model_id"]
                    provider_fn = PROVIDER_FUNCTIONS[provider]

                    pbar.set_postfix(
                        prompt=prompt["prompt_id"],
                        model=model_name,
                    )

                    # Query the model
                    response = query_with_retry(
                        provider_fn, prompt["prompt_text"], model_id
                    )

                    error = ""
                    if response.startswith("[ERROR]"):
                        error = response
                        response_text = ""
                    else:
                        response_text = response

                    # Write result
                    writer.writerow(
                        {
                            "prompt_id": prompt["prompt_id"],
                            "category": prompt["category"],
                            "difficulty_level": prompt["difficulty_level"],
                            "difficulty_label": prompt["difficulty_label"],
                            "prompt_text": prompt["prompt_text"],
                            "expected_artifact": prompt["expected_artifact"],
                            "model_name": model_name,
                            "provider": provider,
                            "response_text": response_text,
                            "response_length": len(response_text),
                            "timestamp": datetime.now().isoformat(),
                            "error": error,
                        }
                    )
                    f.flush()

                    # Rate limiting
                    time.sleep(RATE_LIMIT_DELAY.get(provider, 1.0))
                    pbar.update(1)

        if skipped > 0:
            print(f"Skipped {skipped} already-completed tasks.")

    print(f"\nResults saved to: {RESULTS_FILE}")
    print(f"Total responses: {total_tasks - skipped}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Verify API keys
    missing_keys = []
    if not os.getenv("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY")
    if not os.getenv("ANTHROPIC_API_KEY"):
        missing_keys.append("ANTHROPIC_API_KEY")
    if not os.getenv("GOOGLE_API_KEY"):
        missing_keys.append("GOOGLE_API_KEY")
    if not os.getenv("DEEPSEEK_API_KEY"):
        missing_keys.append("DEEPSEEK_API_KEY")
    if not os.getenv("DASHSCOPE_API_KEY"):
        missing_keys.append("DASHSCOPE_API_KEY")

    if missing_keys:
        print("WARNING: Missing API keys:")
        for key in missing_keys:
            print(f"  - {key}")
        print("Set them in a .env file or as environment variables.")
        print("Continuing with available providers only...\n")

        # Remove models with missing keys
        key_to_provider = {
            "OPENAI_API_KEY": "openai",
            "ANTHROPIC_API_KEY": "anthropic",
            "GOOGLE_API_KEY": "google",
            "DEEPSEEK_API_KEY": "deepseek",
            "DASHSCOPE_API_KEY": "dashscope",
        }
        for key in missing_keys:
            provider = key_to_provider[key]
            to_remove = [
                m for m, c in MODELS.items() if c["provider"] == provider
            ]
            for m in to_remove:
                del MODELS[m]
                print(f"  Removed model: {m}")

    if not MODELS:
        print("ERROR: No API keys available. Cannot run evaluation.")
        exit(1)

    print(f"Models to evaluate: {list(MODELS.keys())}")
    run_evaluation()
