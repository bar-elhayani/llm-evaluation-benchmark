import anthropic
import pandas as pd
import json
from pathlib import Path
from dotenv import load_dotenv
import time

load_dotenv()

QUESTIONS_PATH = "data/questions.csv"
OUTPUT_PATH = "data/synthetic_pairs.jsonl"

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a football (soccer) expert and educator. 
When given a football question, you will generate TWO responses:
1. A STRONG response: accurate, detailed, well-structured, with concrete examples
2. A WEAK response: vague, incomplete, partially incorrect, lacking depth

Return ONLY a JSON object with exactly this structure, no other text:
{
    "strong": "the strong response here",
    "weak": "the weak response here"
}"""

CATEGORY_CONTEXT = {
    "CONCEPT": "Focus on tactical and positional concepts in soccer.",
    "HISTORY": "Focus on historical facts, records, dates, and statistics in soccer.",
    "COMPARE": "Focus on comparing tactics, formations, styles, or players in soccer.",
    "ANALYSIS": "Focus on analytical reasoning about why things happened in soccer.",
}


def generate_pair(question: str, category: str) -> dict | None:
    prompt = f"{CATEGORY_CONTEXT.get(category, '')}\n\nQuestion: {question}"

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1500,
            messages=[
                {"role": "user", "content": prompt}
            ],
            system=SYSTEM_PROMPT,
        )

        raw = message.content[0].text.strip()
        # Clean up any markdown code blocks if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)

        if "strong" not in result or "weak" not in result:
            print(f"    WARNING: missing keys in response")
            return None

        return result

    except json.JSONDecodeError as e:
        print(f"    ERROR parsing JSON: {e}")
        return None
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def build_system_prompt(category: str) -> str:
    category_context = {
        "CONCEPT": "You are a football expert. Explain concepts clearly with accurate definitions and concrete examples.",
        "HISTORY": "You are a football historian. Provide accurate facts, dates, and statistics without hallucination.",
        "COMPARE": "You are a football analyst. Compare options clearly, covering tradeoffs and when each applies.",
        "ANALYSIS": "You are a football analyst. Argue a clear position with supporting evidence and acknowledge complexity.",
    }
    return category_context.get(category, "You are a knowledgeable football expert.")


def main():
    questions_df = pd.read_csv(QUESTIONS_PATH)
    print(f"Loaded {len(questions_df)} questions")
    print(f"Generating synthetic preference pairs using Claude...\n")

    pairs = []
    failed = 0

    for _, row in questions_df.iterrows():
        question_id = row["id"]
        question = row["question"]
        category = row["category"]
        difficulty = row["difficulty"]

        print(f"  {question_id} [{category}]: {question[:60]}...")

        result = generate_pair(question, category)

        if result is None:
            failed += 1
            continue

        pair = {
            "prompt": build_system_prompt(category) + f"\n\nQuestion: {question}",
            "chosen": result["strong"],
            "rejected": result["weak"],
            "metadata": {
                "question_id": question_id,
                "category": category,
                "difficulty": difficulty,
                "chosen_model": "claude-synthetic-strong",
                "rejected_model": "claude-synthetic-weak",
                "chosen_score": 9.0,  # synthetic strong = assumed high quality
                "rejected_score": 3.0,  # synthetic weak = assumed low quality
                "score_diff": 6.0,
                "source": "synthetic",
            }
        }
        pairs.append(pair)
        time.sleep(0.3)  # avoid rate limiting

    print(f"\nGenerated {len(pairs)} synthetic pairs, {failed} failed")

    # Save synthetic pairs
    output_path = Path(OUTPUT_PATH)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Saved to {output_path}")

    # Now merge with original 26 pairs
    original_path = Path("data/preferences.jsonl")
    merged_path = Path("data/preferences_merged.jsonl")

    original_pairs = []
    with open(original_path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            p["metadata"]["source"] = "original"
            original_pairs.append(p)

    all_pairs = original_pairs + pairs

    with open(merged_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nMerged dataset: {len(all_pairs)} total pairs")
    print(f"  Original (score-based): {len(original_pairs)}")
    print(f"  Synthetic (Claude-generated): {len(pairs)}")
    print(f"Saved merged dataset to {merged_path}")

    # Breakdown by category
    by_cat = {}
    for p in all_pairs:
        cat = p["metadata"]["category"]
        by_cat[cat] = by_cat.get(cat, 0) + 1
    print("\nBreakdown by category:")
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat}: {count} pairs")


if __name__ == "__main__":
    main()