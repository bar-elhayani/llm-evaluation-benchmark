import pandas as pd
import json
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
SCORES_PATH     = "data/judge_scores.csv"
RESPONSES_PATH  = "data/responses.csv"
QUESTIONS_PATH  = "data/questions.csv"
OUTPUT_PATH     = "data/preferences.jsonl"
THRESHOLD       = 2.0
# ─────────────────────────────────────────────────────────────────────────────

def build_system_prompt(category: str) -> str:
    category_context = {
        "CONCEPT":  "You are a football expert. Explain concepts clearly with accurate definitions and concrete examples.",
        "HISTORY":  "You are a football historian. Provide accurate facts, dates, and statistics without hallucination.",
        "COMPARE":  "You are a football analyst. Compare options clearly, covering tradeoffs and when each applies.",
        "ANALYSIS": "You are a football analyst. Argue a clear position with supporting evidence and acknowledge complexity.",
    }
    return category_context.get(category, "You are a knowledgeable football expert.")


def build_preference_pairs(scores_df, responses_df, questions_df):
    pairs = []
    skipped = 0

    # Build a lookup: question_id → actual question text
    question_lookup = dict(zip(questions_df["id"], questions_df["question"]))

    merged = responses_df.merge(
        scores_df[["question_id", "model", "judge_score", "judge_explanation"]],
        on=["question_id", "model"],
        how="left"
    )

    for question_id, group in merged.groupby("question_id"):
        if len(group) < 2:
            continue

        rows = {row["model"]: row for _, row in group.iterrows()}
        models = list(rows.keys())

        if len(models) != 2:
            continue

        model_a, model_b = models[0], models[1]
        score_a = rows[model_a]["judge_score"]
        score_b = rows[model_b]["judge_score"]

        diff = score_a - score_b
        abs_diff = abs(diff)

        if abs_diff < THRESHOLD:
            skipped += 1
            continue

        if diff > 0:
            chosen_model, rejected_model = model_a, model_b
            chosen_score, rejected_score = score_a, score_b
        else:
            chosen_model, rejected_model = model_b, model_a
            chosen_score, rejected_score = score_b, score_a

        chosen_row   = rows[chosen_model]
        rejected_row = rows[rejected_model]
        category     = chosen_row["category"]

        # Look up the real question text
        question_text = question_lookup.get(question_id, f"[missing: {question_id}]")

        pair = {
            "prompt": build_system_prompt(category) + f"\n\nQuestion: {question_text}",
            "chosen": chosen_row["response"],
            "rejected": rejected_row["response"],
            "metadata": {
                "question_id":    question_id,
                "category":       category,
                "difficulty":     str(chosen_row.get("difficulty", "")),
                "chosen_model":   chosen_model,
                "rejected_model": rejected_model,
                "chosen_score":   float(chosen_score),
                "rejected_score": float(rejected_score),
                "score_diff":     float(abs_diff),
            }
        }
        pairs.append(pair)

    print(f"Result: {len(pairs)} pairs kept, {skipped} skipped (noise below threshold {THRESHOLD})")
    return pairs


def main():
    print("Loading data...")
    scores_df    = pd.read_csv(SCORES_PATH)
    responses_df = pd.read_csv(RESPONSES_PATH)
    questions_df = pd.read_csv(QUESTIONS_PATH, quotechar='"', quoting=1, on_bad_lines='skip')

    print(f"  Scores: {len(scores_df)} rows")
    print(f"  Responses: {len(responses_df)} rows")
    print(f"  Questions: {len(questions_df)} rows")
    print(f"  Threshold: {THRESHOLD} points\n")

    pairs = build_preference_pairs(scores_df, responses_df, questions_df)

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Saved {len(pairs)} preference pairs to {output_path}")

    if pairs:
        by_category = {}
        for p in pairs:
            cat = p["metadata"]["category"]
            by_category[cat] = by_category.get(cat, 0) + 1
        print("\nBreakdown by category:")
        for cat, count in sorted(by_category.items()):
            print(f"  {cat}: {count} pairs")


if __name__ == "__main__":
    main()