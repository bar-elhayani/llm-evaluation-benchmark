import csv
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

SYSTEM_PROMPT = "You are a knowledgeable football (soccer) analyst. Answer questions clearly and accurately."
QUESTIONS_FILE = "data/questions.csv"
RESPONSES_FILE = "data/responses.csv"

MODELS = [
    ("groq-llama", "llama-3.1-8b-instant"),
    ("groq-llama70b", "llama-3.3-70b-versatile"),
]


def ask_groq(client: Groq, model_id: str, question: str) -> str:
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


def load_questions(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    questions = load_questions(QUESTIONS_FILE)

    os.makedirs("data", exist_ok=True)
    write_header = not os.path.exists(RESPONSES_FILE)

    with open(RESPONSES_FILE, "a", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(
            out_f,
            fieldnames=["question_id", "category", "difficulty", "model", "response", "timestamp"],
        )
        if write_header:
            writer.writeheader()

        for row in questions:
            qid = row["id"]
            for model_name, model_id in MODELS:
                print(f"Running {qid} on {model_name}...")
                response_text = ask_groq(client, model_id, row["question"])
                writer.writerow(
                    {
                        "question_id": qid,
                        "category": row["category"],
                        "difficulty": row["difficulty"],
                        "model": model_name,
                        "response": response_text,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                out_f.flush()
                time.sleep(1)


if __name__ == "__main__":
    main()
