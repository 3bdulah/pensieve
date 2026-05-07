"""
build_index.py — One-time script to build both FAISS indexes from the Excel data.

Run this once before starting the app, and again any time the data file changes.

Output files:
  retrieval/faiss_questions.index      — Index 1: questions only (for direct answer lookup)
  retrieval/faiss_fulldata.index       — Index 2: all content   (for context retrieval)
  retrieval/vectorizer_questions.skops — TF-IDF vectorizer for Index 1
  retrieval/vectorizer_fulldata.skops  — TF-IDF vectorizer for Index 2
  data/questions.json                  — Question strings for Index 1
  data/answers.json                    — Answer strings for Index 1
  data/all_texts.json                  — All text chunks for Index 2
"""

import pandas as pd
import faiss
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from skops.io import dump

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_excel("data/harry_potter_data_02.xlsx")

# Rows with a non-null 'answer' column are Q&A pairs (used by both indexes)
# Rows without an answer are raw informational chunks (used only by Index 2)
qa_df = df[df["answer"].notna()].reset_index(drop=True)
all_texts = df["content"].fillna("").tolist()
questions = qa_df["content"].tolist()
answers = qa_df["answer"].tolist()

print(f"Total rows: {len(df)} | QA pairs: {len(qa_df)} | Raw info: {len(df) - len(qa_df)}")

# ── FAISS Index 1: Questions only ──────────────────────────────────────────────
# TF-IDF with bigrams captures Harry Potter phrase patterns.
# L2 normalization + IndexFlatIP (inner product) gives cosine similarity search.
print("\nBuilding FAISS index 1 (questions only)...")
vec1 = TfidfVectorizer(ngram_range=(1, 2))
q_embeddings = vec1.fit_transform(questions).toarray().astype("float32")
q_embeddings = normalize(q_embeddings, norm="l2")

index1 = faiss.IndexFlatIP(q_embeddings.shape[1])
index1.add(q_embeddings)

faiss.write_index(index1, "retrieval/faiss_questions.index")
with open("data/questions.json", "w", encoding="utf-8") as f:
    json.dump(questions, f, ensure_ascii=False)
with open("data/answers.json", "w", encoding="utf-8") as f:
    json.dump(answers, f, ensure_ascii=False)
dump(vec1, "retrieval/vectorizer_questions.skops")
print(f"  Indexed {len(questions)} questions")

# ── FAISS Index 2: Full knowledge base ────────────────────────────────────────
# Trained on all content (questions + answers + raw info).
# At query time, the top N most similar chunks are retrieved and injected
# into the LLM system prompt as grounding context.
print("\nBuilding FAISS index 2 (full data)...")
vec2 = TfidfVectorizer(ngram_range=(1, 2))
all_embeddings = vec2.fit_transform(all_texts).toarray().astype("float32")
all_embeddings = normalize(all_embeddings, norm="l2")

index2 = faiss.IndexFlatIP(all_embeddings.shape[1])
index2.add(all_embeddings)

faiss.write_index(index2, "retrieval/faiss_fulldata.index")
with open("data/all_texts.json", "w", encoding="utf-8") as f:
    json.dump(all_texts, f, ensure_ascii=False)
dump(vec2, "retrieval/vectorizer_fulldata.skops")
print(f"  Indexed {len(all_texts)} chunks")

print("\nAll indexes built successfully!")
