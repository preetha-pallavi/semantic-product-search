# Semantic Product Search

Search Amazon products by meaning, not keywords — built with sentence embeddings, FAISS vector search, FastAPI, and Streamlit.

## Demo

[▶️ Watch the 5-second semantic search demo](live_demo/semantic_search_demo.mp4)

## What it does

Instead of matching exact keywords, this app understands what people actually said in product reviews. Search "aloe vera" and it finds products whose reviews describe aloe-related content — even if the word "aloe" isn't in the product name.

**Example:** searching "matcha" returns matcha teas, matcha-flavored candy, and related products, ranked by how closely their reviews match the query.

## Dataset

[Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) — ~568K reviews across ~74K products. After cleaning (deduplication, filtering low-signal products/reviews), the working dataset is ~342K reviews across ~27.6K products.

## How it works

1. **Clean the data** — remove duplicate reviews, drop products with too few reviews, strip HTML artifacts.

2. **Generate embeddings** — each review is converted into a vector using `sentence-transformers` (`all-MiniLM-L6-v2`), then averaged per product.

3. **Build a search index** — all product vectors are indexed with FAISS for fast similarity search.

4. **Serve it** — a FastAPI backend exposes search and recommendation endpoints; a Streamlit app provides the interface.

## FAISS Index

The project uses `product_index.faiss` as the precomputed FAISS vector index for semantic product search.

The index contains the product-level embeddings generated from the review data and is loaded by the FastAPI backend at startup. It is required for the API to perform semantic similarity search.

The index is generated in `notebooks/embedding.ipynb` and stored as `product_index.faiss` so the application can run without rebuilding the embeddings every time.

## Match quality

Instead of showing raw similarity scores, results are grouped into three intuitive tiers:

* **Strong match** — very similar reviews
* **Good match** — related, but less precise
* **Similar** — a looser connection

This makes results easier to interpret without needing to understand the underlying similarity metric.

## Notebooks

| Notebook                             | What it does                                                                                                                                                                                                       |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `notebooks/EDA.ipynb`                | Loads the raw dataset, inspects nulls/duplicates/distributions, cleans it (dedupe, filter low-review products, strip HTML), saves the cleaned dataset as Parquet                                                   |
| `notebooks/embedding.ipynb`          | Generates review embeddings, aggregates them per product, builds and saves the FAISS index, evaluates recommendation quality with Recall@k                                                                         |
| `notebooks/summarization_demo.ipynb` | Exploratory notebook testing different approaches for generating human-readable product labels (AI summarization, review summaries) — kept for reference; the final approach (TF-IDF keywords) lives in `tfidf.py` |

## Scripts

| File       | What it does                                                                                                                                                         |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tfidf.py` | Generates short keyword-based labels for each product (e.g. `matcha, green tea, organic`) using TF-IDF over review text — used to make search results human-readable |
| `api.py`   | FastAPI backend — exposes `/search` (free-text semantic search) and `/recommend` (item-to-item similarity) endpoints                                                 |
| `demo.py`  | Streamlit frontend — the search interface users interact with, calls the API and displays results                                                                    |

## Evaluation

Since there's no single "accuracy" metric for a recommender, this project uses:

* **Qualitative review** — manually inspecting recommendations for topical coherence
* **Recall@k** — using pairs of products the same user reviewed as a weak proxy for relatedness

| k  | Recall@k |
| -- | -------: |
| 5  |    0.106 |
| 10 |    0.142 |
| 20 |    0.176 |

This is a content-only recommender (no purchase history, no collaborative signal) — these numbers substantially beat random chance (~0.036%).

## Tools used

* **sentence-transformers** — turns review text into semantic vectors
* **FAISS** — fast similarity search over product vectors
* **scikit-learn** — TF-IDF keyword extraction for product labels
* **FastAPI** — backend API (`/search`, `/recommend`)
* **Streamlit** — interactive search demo
* **Docker / Docker Compose** — containerizes the API and UI as separate services for consistent, portable deployment
* **uv** — Python dependency management

## Running it

### With Docker (recommended)

```bash
docker compose up --build
```

Then open `http://localhost:8501`.

### Locally (without Docker)

```bash
uv sync

uvicorn api:app --reload          # terminal 1
streamlit run demo.py             # terminal 2
```

## Project structure

```text
├── notebooks/
│   ├── EDA.ipynb                    # data cleaning
│   ├── embedding.ipynb              # embeddings + FAISS index + evaluation
│   └── summarization_demo.ipynb
├── live_demo/
│   └── semantic_search_demo.mp4     # short demo video
├── product_index.faiss              # precomputed FAISS product index
├── tfidf.py                          # generates product keyword labels
├── api.py                            # FastAPI backend
├── demo.py                           # Streamlit frontend
├── Dockerfile.api
├── Dockerfile.streamlit
├── docker-compose.yml
└── pyproject.toml
```

## A known limitation

This dataset has no product-name field — only product IDs and reviews. Several approaches were tried to generate human-readable labels (AI-generated summaries, review titles, TF-IDF keywords); the final version uses TF-IDF keywords as the closest approximation to a descriptive label. In a production setting, this would be paired with real product metadata.
