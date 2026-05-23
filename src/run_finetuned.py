import requests
import pandas as pd
import time
from pathlib import Path

# Ollama runs locally on port 11434
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "football-dpo-v2"
QUESTIONS_PATH = "data/questions.csv"
OUTPUT_PATH = "data/responses_finetuned_v2.csv"

SYSTEM_PROMPT = "You are an expert in association football (soccer). All questions are about soccer/football as played in Europe, South America, and worldwide — not American football. Provide accurate, detailed answers about tactics, history, players, and formations."

def query_ollama(question: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": question,
        "system": SYSTEM_PROMPT,
        "stream": False,
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["response"]


def main():
    questions_df = pd.read_csv(QUESTIONS_PATH)
    print(f"Loaded {len(questions_df)} questions")

    results = []
    for _, row in questions_df.iterrows():
        question_id = row["id"]
        question    = row["question"]
        category    = row["category"]
        difficulty  = row["difficulty"]

        print(f"  Running {question_id} [{category}]...")

        try:
            response = query_ollama(question)
            results.append({
                "question_id": question_id,
                "category":    category,
                "difficulty":  difficulty,
                "model":       MODEL_NAME,
                "response":    response,
            })
        except Exception as e:
            print(f"  ERROR on {question_id}: {e}")
            results.append({
                "question_id": question_id,
                "category":    category,
                "difficulty":  difficulty,
                "model":       MODEL_NAME,
                "response":    f"ERROR: {e}",
            })

        time.sleep(0.5)

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nSaved {len(results)} responses to {output_path}")


if __name__ == "__main__":
    main()