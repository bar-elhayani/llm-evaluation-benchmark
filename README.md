# LLM Evaluation Benchmark - Football Knowledge

## Overview

This project builds a full automated evaluation pipeline that tests two LLMs on football knowledge questions across 4 categories and 3 difficulty levels. Responses are scored by Claude Sonnet acting as an LLM-as-judge using a custom rubric tailored to each question type, removing the need for manual evaluation. The project then extends the pipeline with a RAG system that retrieves Wikipedia context at query time, and systematically compares RAG vs baseline performance across models and categories. Finally, a live Streamlit demo lets users ask football questions powered by the full RAG pipeline, providing real-time answers grounded in Wikipedia content.

## Live Demo

[Football Knowledge Assistant](llm-evaluation-benchmark-hczxhdkbrskgj5d3kautkn.streamlit.app)

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

## Evaluation Method

Instead of manual scoring, Claude Sonnet was used as an LLM-as-judge. Each response was sent to Claude alongside a structured rubric that defined exactly what a good answer must include, per category. All categories are scored out of 10 points.

| Category | Rubric |
|----------|--------|
| **CONCEPT** | Accurate definition (3pts) + concrete example (3pts) + simple language (2pts) + when/why to use it (2pts) |
| **HISTORY** | Correct facts including exact numbers and dates (5pts) + relevant context (3pts) + no hallucinated stats (2pts) |
| **COMPARE** | Clear explanation of each side (3pts) + direct comparison (4pts) + when each is preferred (3pts) |
| **ANALYSIS** | Clear main argument (3pts) + 2 supporting reasons with real examples (4pts) + acknowledgment of complexity (3pts) |

## Question Categories

| Category | Description |
|----------|-------------|
| **CONCEPT** | Tactical and positional concepts |
| **HISTORY** | Facts, records, dates, statistics |
| **COMPARE** | Comparing formations, styles, players |
| **ANALYSIS** | Reasoning about why things happened |

## Key Findings

### Baseline Results

- **Llama 3.3 70B wins overall: 7.43 vs 6.01**
- 70B wins in every single category
- Biggest gap: **HISTORY** (1.6 pts) - model size matters most for factual recall
- Smallest gap: **ANALYSIS** (0.7 pts) - both models struggle similarly with complex reasoning
- Interpretation: for reasoning tasks, the smaller and cheaper 8B model is nearly competitive with 70B

### RAG Results

- **RAG hurt both models**
- Llama 3.1 8B: 6.01 → 5.76 (−0.25)
- Llama 3.3 70B: 7.43 → 6.60 (−0.83)
- Most affected category: **CONCEPT** (−0.83 pts)
- **Llama 3.1 8B handled RAG better** (smaller performance drop)

### Why did RAG hurt performance? (Deep Dive)

Naive RAG introduced four distinct failure modes in this benchmark. First, **chunk size was too small** - at 500 characters per chunk, retrieved passages were often too short to contain a complete answer, adding partial context that confused the model rather than helping it. Second, **models over-relied on retrieved text** - both models appeared to prioritize the retrieved chunks over their own training knowledge, even when the chunks were only partially relevant to the question. Third, **the larger model was hurt more** - the 70B model has stronger internal knowledge built up during training, so when noisy or incomplete context conflicted with what it already knew, it caused more confusion; the 8B model has weaker priors and was therefore less destabilized by added context. Fourth, **Wikipedia retrieval mismatch** - some questions required synthesizing information across multiple topics, but the retrieval system returned only one page, missing important context that would have been needed for a complete answer.

### What this tells us

Naive RAG is not always better than no RAG. The quality of retrieval, chunk size, and the model's existing knowledge all interact to determine whether RAG helps or hurts. A stronger model that already knows the answer can be made worse by noisy context, while a weaker model has less to lose. This is a well-documented challenge in production RAG systems, and highlights the importance of evaluating RAG empirically rather than assuming it will always improve results.

## Tech Stack

| Tool | Why it was used |
|------|-----------------|
| **Python** | Pipeline orchestration and scripting |
| **Groq API** | Fast LLM inference for Llama models; chosen for free tier and speed |
| **Anthropic API** | Claude Sonnet used as LLM-as-judge for automated scoring |
| **FAISS** | Vector database for storing and retrieving embedded text chunks; enables fast similarity search at scale |
| **sentence-transformers** (`all-MiniLM-L6-v2`) | Lightweight embedding model that converts text chunks and questions into vectors for semantic similarity search |
| **wikipedia-api** | Real-time Wikipedia retrieval for the Streamlit demo |
| **pandas** | Data manipulation and analysis across all CSV files |
| **matplotlib** | Generating result charts saved to the `results/` folder |
| **Streamlit** | Interactive demo app with live RAG pipeline |
| **Jupyter** | Interactive analysis notebook with 6 charts and full findings |

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

# 11. Run the Streamlit demo
streamlit run src/app.py
```
