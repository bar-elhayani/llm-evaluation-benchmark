import csv
import os
import time

import anthropic
from dotenv import load_dotenv

load_dotenv()

RESPONSES_FILE = "data/responses.csv"
QUESTIONS_FILE = "data/questions.csv"
SCORES_FILE = "data/judge_scores.csv"

RUBRICS = {
    "CONCEPT": (
        "- Accurate and clear definition of the concept (3 points)\n"
        "- A concrete example of how it works in a real match or team (3 points)\n"
        "- Simple language that a non-expert could understand (2 points)\n"
        "- Mention of when/why a team would use it (2 points)"
    ),
    "HISTORY": (
        "- Correct and precise facts including exact numbers and dates (5 points)\n"
        "- Relevant context around the fact (3 points)\n"
        "- No hallucinated statistics or names (2 points)"
    ),
    "COMPARE": (
        "- Clear explanation of each side separately (3 points)\n"
        "- Direct comparison with specific differences (4 points)\n"
        "- Mention of when each option is preferred (3 points)"
    ),
    "ANALYSIS": (
        "- A clear main argument/thesis (3 points)\n"
        "- At least 2 supporting reasons with a specific real-world example (match, season, or player) (4 points)\n"
        "- Acknowledgment of complexity or alternative views (3 points)"
    ),
}

JUDGE_PROMPT = """\
You are an expert football (soccer) analyst and evaluator. Score the following response to a football question.

Question: {question}

Response to evaluate:
{response}

Scoring rubric (total: 10 points):
{rubric}

Instructions:
- Award points based strictly on the rubric criteria above.
- Return your evaluation in exactly this format (two lines, nothing else):
SCORE: <integer from 1 to 10>
EXPLANATION: <one or two sentences explaining the score>"""


def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def judge(client: anthropic.Anthropic, question: str, response: str, category: str) -> tuple[int, str]:
    prompt = JUDGE_PROMPT.format(
        question=question,
        response=response,
        rubric=RUBRICS[category],
    )
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    text = message.content[0].text.strip()

    score_line, explanation_line = "", ""
    for line in text.splitlines():
        if line.startswith("SCORE:"):
            score_line = line.split(":", 1)[1].strip()
        elif line.startswith("EXPLANATION:"):
            explanation_line = line.split(":", 1)[1].strip()

    score = int(score_line) if score_line.isdigit() else 0
    return score, explanation_line


def main():
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    responses = load_csv(RESPONSES_FILE)
    questions = {row["id"]: row["question"] for row in load_csv(QUESTIONS_FILE)}

    write_header = not os.path.exists(SCORES_FILE)

    with open(SCORES_FILE, "a", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(
            out_f,
            fieldnames=["question_id", "category", "difficulty", "model", "judge_score", "judge_explanation"],
        )
        if write_header:
            writer.writeheader()

        for row in responses:
            qid = row["question_id"]
            model = row["model"]
            category = row["category"]
            print(f"Judging {qid} - {model}...")

            score, explanation = judge(
                client,
                question=questions[qid],
                response=row["response"],
                category=category,
            )
            writer.writerow(
                {
                    "question_id": qid,
                    "category": category,
                    "difficulty": row["difficulty"],
                    "model": model,
                    "judge_score": score,
                    "judge_explanation": explanation,
                }
            )
            out_f.flush()
            time.sleep(0.5)


if __name__ == "__main__":
    main()
