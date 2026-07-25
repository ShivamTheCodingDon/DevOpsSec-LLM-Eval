import json
import glob
import os
import argparse

parser = argparse.ArgumentParser(description="Generate summary for scored results")
parser.add_argument("--input", type=str, default=None, help="Path to the scored responses JSON file")
args = parser.parse_args()

if args.input:
    latest_file = args.input
else:
    # Recursively find scored_responses files in results/
    files = glob.glob(r'd:\000Coding\ACM\results\**\scored_responses_*.json', recursive=True)
    if not files:
        # Fallback to base directory
        files = glob.glob(r'd:\000Coding\ACM\results\scored_responses_*.json')
        
    if not files:
        print("No scored_responses JSON files found.")
        exit()
    latest_file = max(files, key=os.path.getctime)

with open(latest_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

results = data.get('results', [])
total = len(results)

if total == 0:
    print("No results found.")
    exit()

# Academically rigorous handling of API blocks:
# If the response was blocked by content filters (-1), count it as a Platform Refusal (refusal=1, harm=0)
for r in results:
    scores = r.get('judge_scores', {})
    if scores.get('harmfulness', -1) == -1:
        scores['harmfulness'] = 0
        scores['refusal'] = 1
    r['judge_scores'] = scores

refusals = sum(1 for r in results if r.get('judge_scores', {}).get('refusal') == 1)
harmfulness = [r.get('judge_scores', {}).get('harmfulness', 0) for r in results]
avg_harm = sum(harmfulness) / total

print(f"--- ACADEMIC EVALUATION SUMMARY ({data.get('metadata', {}).get('model', 'Unknown')}) ---")
print(f"Total Prompts: {total}")
print(f"Overall Refusal Rate: {refusals/total*100:.1f}% ({refusals}/{total})")
print(f"Average Harmfulness Score (0-5): {avg_harm:.2f}")

# By Difficulty
diff_stats = {}
for r in results:
    d = r.get('difficulty_level', 'unknown')
    score = r.get('judge_scores', {}).get('harmfulness', 0)
    refusal = r.get('judge_scores', {}).get('refusal', 0)
    if d not in diff_stats:
        diff_stats[d] = {'count': 0, 'harm': 0, 'refusals': 0}
    diff_stats[d]['count'] += 1
    diff_stats[d]['harm'] += score
    diff_stats[d]['refusals'] += refusal

print("\n--- By Difficulty Level ---")
for d in sorted(diff_stats.keys()):
    st = diff_stats[d]
    c = st['count']
    if c > 0:
        print(f"Level {d} ({c} prompts): Refusals = {st['refusals']/c*100:.1f}%, Avg Harmfulness = {st['harm']/c:.2f}")

# By Category
cat_stats = {}
for r in results:
    cat = r.get('category', 'unknown')
    score = r.get('judge_scores', {}).get('harmfulness', 0)
    refusal = r.get('judge_scores', {}).get('refusal', 0)
    if cat not in cat_stats:
        cat_stats[cat] = {'count': 0, 'harm': 0, 'refusals': 0}
    cat_stats[cat]['count'] += 1
    cat_stats[cat]['harm'] += score
    cat_stats[cat]['refusals'] += refusal

print("\n--- By Category ---")
for cat, st in sorted(cat_stats.items(), key=lambda x: x[1]['harm']/x[1]['count'], reverse=True):
    c = st['count']
    if c > 0:
        print(f"{cat:<22} | Harmfulness: {st['harm']/c:.2f} | Refusals: {st['refusals']/c*100:>5.1f}%")

