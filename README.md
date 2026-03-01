# Omnilex Agentic Legal Retrieval

Given a legal question in English, retrieve the correct Swiss legal citations from a German/French/Italian corpus. Scored on **Macro F1** — how well your predicted citation sets match the gold citation sets across all test queries.

---

## Setup

```bash
git clone https://github.com/alanyom/AI-ML-SP26/tree/main
cd AI-ML-SP26/Law_Retrieval

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Data

Most data files are included in the repo. Download the two large corpus files from the Kaggle competition page and place them in the `data/` folder:

- `laws_de.csv`
- `court_considerations.csv`

| File | Description |
|------|-------------|
| `train.csv` | Training queries with gold citations (German legal questions, from LEXam) |
| `val.csv` | 10 English validation queries with gold citations |
| `test.csv` | 40 English test queries — no labels, generate predictions for these |
| `laws_de.csv` | Retrieval corpus of Swiss federal law snippets, keyed by canonical citation |
| `court_considerations.csv` | Retrieval corpus of Swiss Federal Court decisions (~30 years), keyed by canonical citation. Note: older decisions are missing and are not expected in gold citations |
| `sample_submission.csv` | Shows the required submission format |

> You can start development with just `laws_de.csv` while `court_considerations.csv` downloads — it's a large file.

---

## Workflow

**1. Build your solution**

Open `notebook.ipynb`. The typical approach:
- Learn patterns from `train.csv`
- Sanity check on `val.csv`
- Search through `laws_de.csv` + `court_considerations.csv` to find relevant citations for each query
- Generate predictions for every `query_id` in `test.csv`
- Save as `submission.csv`

**2. Evaluate locally**

```bash
# Evaluate against val set (default)
python evaluation/evaluate.py submission.csv

# Evaluate against train set
python evaluation/evaluate.py submission.csv --split train

# Evaluate against a custom solution file
python evaluation/evaluate.py submission.csv --solution path/to/solution.csv

# Show per-query breakdown
python evaluation/evaluate.py submission.csv -v
```

**3. Run tests**

```bash
pytest tests/
```

---

## Submission Format

One row per query, with citations semicolon-separated:

```
query_id,predicted_citations
q_001,"SR 210 Art. 1;BGE 116 Ia 56 E. 2b"
q_002,"SR 311.0 Art. 117"
```

The test set has 40 English queries — 20 scored on the public leaderboard, 20 on the private leaderboard.

---

## Scoring

The primary metric is **Macro F1** — per-query F1 between your predicted citation set and the gold citation set, averaged across all queries.

For each query:
- **Precision** = correct citations predicted / all citations predicted
- **Recall** = correct citations predicted / all gold citations
- **F1** = harmonic mean of precision and recall

Final score = average F1 across all test queries.

---

## Project Structure

```
├── evaluation/
│   ├── metrics.py          # Macro F1, Micro F1, MAP, NDCG
│   └── evaluate.py         # CLI scoring script
├── data/                   # All data files go here
├── tests/
│   ├── conftest.py
│   └── test_metrics.py
├── notebook.ipynb          # Your solution goes here
└── requirements.txt
```

---

## Tips

- Questions are in English but the corpus is in German/French/Italian — keyword search alone won't work well
- `val.csv` matches the test distribution better than `train.csv` — use it to tune your approach
- Citations must exactly match the canonical format in the corpus (e.g. `BGE 116 Ia 56 E. 2b`, `Art. 1 OR`)
