import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent.parent / "results"

def load_and_aggregate():
    # model -> dataset -> { 'count': X, 'refusals': Y, 'harm': Z }
    stats = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'refusals': 0, 'harm': 0}))
    
    for json_path in RESULTS_DIR.rglob("scored_responses_*.json"):
        model_name = json_path.parent.name
        
        if model_name.startswith('.'):
            continue
        
        # Determine dataset based on filename
        if "advanced_prompts" in json_path.name:
            dataset_key = "advanced"
        elif "devsecops_prompts" in json_path.name:
            dataset_key = "standard"
        else:
            continue
            
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
                
        results = data.get('results', [])
        
        for r in results:
            scores = r.get('judge_scores', {})
            # Academically rigorous handling of API blocks:
            # If the response was blocked by content filters (-1) or missing, count it as a Platform Refusal
            h = scores.get('harmfulness')
            if h is None or h == -1:
                h = 0
                ref = 1
            else:
                ref = scores.get('refusal', 0)
                if ref is None or ref == -1:
                    ref = 0
            
            stats[model_name][dataset_key]['count'] += 1
            stats[model_name][dataset_key]['harm'] += h
            stats[model_name][dataset_key]['refusals'] += ref
            
            # Add to combined
            stats[model_name]['combined']['count'] += 1
            stats[model_name]['combined']['harm'] += h
            stats[model_name]['combined']['refusals'] += ref

    return stats

def generate_markdown(stats):
    output_file = RESULTS_DIR / "paper_summary_table.md"
    
    models = sorted(stats.keys(), key=lambda x: x.lower())
    
    lines = []
    lines.append("# LLM Security Evaluation Summary")
    lines.append("")
    lines.append("This table contains the aggregated safety metrics (Refusal Rate and Average Harmfulness) across all evaluated models.")
    lines.append("")
    lines.append("| Model | Standard Refusal Rate | Standard Avg Harmfulness | Advanced Refusal Rate | Advanced Avg Harmfulness | Combined Refusal Rate | Combined Avg Harmfulness |")
    lines.append("|-------|-----------------------|--------------------------|-----------------------|--------------------------|-----------------------|--------------------------|")
    
    for model in models:
        st = stats[model]
        
        row = [f"**{model}**"]
        
        for ds in ['standard', 'advanced', 'combined']:
            d_stats = st.get(ds)
            if d_stats and d_stats['count'] > 0:
                c = d_stats['count']
                refusal_rate = (d_stats['refusals'] / c) * 100
                avg_harm = d_stats['harm'] / c
                row.append(f"{refusal_rate:.1f}%")
                row.append(f"{avg_harm:.2f}")
            else:
                row.append("N/A")
                row.append("N/A")
                
        lines.append("| " + " | ".join(row) + " |")
        
    markdown_content = "\n".join(lines)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
        
    print(f"Summary table successfully generated at: {output_file}")
    print("\nPreview:\n")
    print(markdown_content)

if __name__ == "__main__":
    stats = load_and_aggregate()
    generate_markdown(stats)
