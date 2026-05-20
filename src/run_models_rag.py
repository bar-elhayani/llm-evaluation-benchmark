import csv
import os
import time

import faiss
import numpy as np
from dotenv import load_dotenv
from groq import Groq, RateLimitError
from sentence_transformers import SentenceTransformer

load_dotenv()

KNOWLEDGE_BASE_DIR = "data/knowledge_base"
QUESTIONS_FILE = "data/questions.csv"
RESPONSES_FILE = "data/responses_rag.csv"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
EMBED_MODEL = "all-MiniLM-L6-v2"

MODELS = [
    ("groq-llama-rag", "llama-3.1-8b-instant"),
    ("groq-llama70b-rag", "llama-3.3-70b-versatile"),
]

PROMPT_TEMPLATE = """You are a knowledgeable football analyst. Use the following context to answer the question accurately.

Context:
{context}

Question:
{question}"""


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks


def load_knowledge_base(directory: str) -> tuple[list[str], list[str]]:
    chunks, sources = [], []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(directory, filename)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for chunk in chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append(chunk)
            sources.append(filename)
    return chunks, sources


def build_index(chunks: list[str], model: SentenceTransformer):
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def retrieve(question: str, model: SentenceTransformer, index, chunks: list[str]) -> str:
    q_emb = model.encode([question], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_emb)
    _, indices = index.search(q_emb, TOP_K)
    return "\n\n---\n\n".join(chunks[i] for i in indices[0])


def ask_groq(client: Groq, model_id: str, prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
        )
    except RateLimitError:
        print("  Rate limited — waiting 10s before retry...")
        time.sleep(10)
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
        )
    return response.choices[0].message.content


def load_questions(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    print("Loading knowledge base...")
    chunks, _ = load_knowledge_base(KNOWLEDGE_BASE_DIR)
    print(f"  {len(chunks)} chunks loaded from {KNOWLEDGE_BASE_DIR}")

    print("Building embeddings and FAISS index...")
    embed_model = SentenceTransformer(EMBED_MODEL)
    index, _ = build_index(chunks, embed_model)
    print(f"  Index built ({index.ntotal} vectors)")

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    questions = load_questions(QUESTIONS_FILE)

    os.makedirs("data", exist_ok=True)
    write_header = not os.path.exists(RESPONSES_FILE)

    done: set[tuple[str, str]] = set()
    if not write_header:
        with open(RESPONSES_FILE, newline="", encoding="utf-8") as f:
            for existing in csv.DictReader(f):
                done.add((existing["question_id"], existing["model"]))
        print(f"  Skipping {len(done)} already-saved rows")

    with open(RESPONSES_FILE, "a", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(
            out_f,
            fieldnames=["question_id", "category", "difficulty", "model", "response"],
        )
        if write_header:
            writer.writeheader()

        for row in questions:
            qid = row["id"]
            context = retrieve(row["question"], embed_model, index, chunks)
            prompt = PROMPT_TEMPLATE.format(context=context, question=row["question"])

            for model_name, model_id in MODELS:
                if (qid, model_name) in done:
                    continue
                short = model_id.split("-instant")[0].split("-versatile")[0]
                print(f"RAG {qid} - {short}...")
                response_text = ask_groq(client, model_id, prompt)
                writer.writerow(
                    {
                        "question_id": qid,
                        "category": row["category"],
                        "difficulty": row["difficulty"],
                        "model": model_name,
                        "response": response_text,
                    }
                )
                out_f.flush()
                time.sleep(3)

    print("\nDone. Results saved to", RESPONSES_FILE)


if __name__ == "__main__":
    main()
