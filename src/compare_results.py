import pandas as pd

# Load original scores (8B and 70B)
original = pd.read_csv("data/judge_scores.csv")

# Load fine-tuned model scores
finetuned = pd.read_csv("data/judge_scores_finetuned_v2.csv")

print("=" * 60)
print("RESULTS COMPARISON")
print("=" * 60)

# Overall averages
orig_8b  = original[original["model"] == "groq-llama"]["judge_score"].mean()
orig_70b = original[original["model"] == "groq-llama70b"]["judge_score"].mean()
ft_avg   = finetuned["judge_score"].mean()

print(f"\nOverall Average Scores:")
print(f"  Llama 3.1 8B  (baseline):    {orig_8b:.2f}")
print(f"  Llama 3.3 70B (baseline):    {orig_70b:.2f}")
print(f"  football-dpo  (fine-tuned):  {ft_avg:.2f}")

# By category
print(f"\nBy Category:")
categories = ["CONCEPT", "HISTORY", "COMPARE", "ANALYSIS"]

print(f"{'Category':<12} {'8B':>8} {'70B':>8} {'DPO':>8} {'DPO vs 8B':>12} {'DPO vs 70B':>12}")
print("-" * 60)

for cat in categories:
    s_8b  = original[(original["model"] == "groq-llama")    & (original["category"] == cat)]["judge_score"].mean()
    s_70b = original[(original["model"] == "groq-llama70b") & (original["category"] == cat)]["judge_score"].mean()
    s_ft  = finetuned[finetuned["category"] == cat]["judge_score"].mean()

    diff_8b  = s_ft - s_8b
    diff_70b = s_ft - s_70b

    sign_8b  = "+" if diff_8b  >= 0 else ""
    sign_70b = "+" if diff_70b >= 0 else ""

    print(f"{cat:<12} {s_8b:>8.2f} {s_70b:>8.2f} {s_ft:>8.2f} {sign_8b}{diff_8b:>11.2f} {sign_70b}{diff_70b:>11.2f}")

# By difficulty
print(f"\nBy Difficulty:")
difficulties = ["easy", "medium", "hard"]

print(f"{'Difficulty':<12} {'8B':>8} {'70B':>8} {'DPO':>8} {'DPO vs 8B':>12}")
print("-" * 52)

for diff in difficulties:
    s_8b = original[(original["model"] == "groq-llama")    & (original["difficulty"] == diff)]["judge_score"].mean()
    s_70b= original[(original["model"] == "groq-llama70b") & (original["difficulty"] == diff)]["judge_score"].mean()
    s_ft = finetuned[finetuned["difficulty"] == diff]["judge_score"].mean()

    d = s_ft - s_8b
    sign = "+" if d >= 0 else ""
    print(f"{diff:<12} {s_8b:>8.2f} {s_70b:>8.2f} {s_ft:>8.2f} {sign}{d:>11.2f}")

print("\n" + "=" * 60)
print("TRAINING DATA COVERAGE")
print("=" * 60)
print(f"\nQuestions in training pairs by category:")

import json
prefs = []
with open("data/preferences.jsonl") as f:
    prefs = [json.loads(l) for l in f]

coverage = {}
for p in prefs:
    cat = p["metadata"]["category"]
    coverage[cat] = coverage.get(cat, 0) + 1

for cat in categories:
    count = coverage.get(cat, 0)
    total = len(original[original["category"] == cat]) // 2
    print(f"  {cat}: {count}/{total} questions had clear preference signal")