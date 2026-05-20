# LLM Evaluation Benchmark - Football Knowledge

## Overview

This project builds an automated evaluation pipeline that tests two LLMs on football knowledge questions, scores their responses using Claude as an LLM-as-judge, and compares performance with and without RAG (Retrieval-Augmented Generation). Questions span four categories - tactical concepts, historical facts, comparisons, and analytical reasoning - at three difficulty levels. A Wikipedia-sourced knowledge base is used to evaluate whether retrieval-augmented prompting improves or hurts model accuracy.

## Pipeline

**Baseline:**
```
questions.csv → run_models.py → responses.csv → evaluate.py → judge_scores.csv → analysis.ipynb
```

**RAG:**
```
knowledge_base/ → run_models_rag.py → responses_rag.csv → evaluate.py → judge_scores_rag.csv
```

## Models Tested

- **Llama 3.1 8B** (`llama-3.1-8b-instant`) via Groq
- **Llama 3.3 70B** (`llama-3.3-70b-versatile`) via Groq

## Question Categories

| Category | Description |
|----------|-------------|
| **CONCEPT** | Tactical and positional concepts |
| **HISTORY** | Facts, records, dates, statistics |
| **COMPARE** | Comparing formations, styles, players |
| **ANALYSIS** | Reasoning about why things happened |

## Key Findings

### Baseline (no RAG)

- **Llama 3.3 70B wins overall: 7.43 vs 6.01**
- 70B wins in every category
- Biggest gap: **HISTORY** (1.6 pts)
- Smallest gap: **ANALYSIS** (0.7 pts)

### RAG Results

- **RAG hurt both models**
- Llama 3.1 8B: 6.01 → 5.76 (−0.25)
- Llama 3.3 70B: 7.43 → 6.60 (−0.83)
- Most affected category: **CONCEPT**
- **Llama 3.1 8B handled RAG better** (smaller performance drop)

## How to Run

```bash
# 1. Clone the repo
git clone <repo-url>
cd llm-evaluation-benchmark

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add API keys to a .env file
echo "GROQ_API_KEY=your_groq_key" >> .env
echo "ANTHROPIC_API_KEY=your_anthropic_key" >> .env

# 5. Run baseline models
python src/run_models.py

# 6. Evaluate baseline responses
python src/evaluate.py

# 7. Build the knowledge base
python src/build_knowledge_base.py

# 8. Run RAG models
python src/run_models_rag.py

# 9. Evaluate RAG responses
python src/evaluate.py --input data/responses_rag.csv --output data/judge_scores_rag.csv

# 10. Open the analysis notebook
jupyter notebook notebooks/analysis.ipynb
```

## Tech Stack

- **Python** - pipeline scripts
- **Groq API** - LLM inference (Llama models)
- **Anthropic API** - Claude as LLM-as-judge
- **FAISS** - vector similarity search for RAG
- **sentence-transformers** - text embeddings (`all-MiniLM-L6-v2`)
- **pandas** - data manipulation
- **matplotlib** - result charts
- **Jupyter** - interactive analysis notebook
