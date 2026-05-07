# Harry Potter Chatbot – Project Report

| | |
|---|---|
| **University** | Bahcesehir University |
| **Course** | COP4921 – Applied Large Language Models (2025/26) |
| **Professor** | Ahmet Tugrul BAYRAK |
| **Students** | Abdullah Hani Abdellatif Al-Shobaki — 2284612 |
| | Abdulah M. M. Abbas — 2285219 |

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.x |
| Web Framework | Flask 3.1 |
| LLM API | Qwen (qwen-plus) via Alibaba DashScope, OpenAI-compatible endpoint |
| Vector Search | FAISS (faiss-cpu) – IndexFlatIP (cosine similarity) |
| Vectorization | TF-IDF (scikit-learn), bigrams (n-gram range 1–2) |
| Data Storage | Microsoft Excel (.xlsx) via pandas + openpyxl |
| Frontend | HTML/CSS/JavaScript (single-page, no framework) |
| Session Management | Flask server-side sessions |
| Configuration | python-dotenv (.env file) |

---

## System Flow Diagram

```
User types a message in the browser
        │
        ▼
 [Jailbreak Detection]
 is_jailbreak(message)
        │
   YES  │  NO
        │
        ├─YES──► Return "I cannot answer that.."
        │
        ▼
 [FAISS Index 1 – Questions Only]
 TF-IDF vectorize query → cosine similarity search
        │
  score ≥ 0.75?
        │
   YES  │  NO
        │
        ├─YES──► Return stored answer directly (no API call)
        │
        ▼
 [FAISS Index 2 – Full Data]
 TF-IDF vectorize query → retrieve top 5 most similar chunks
        │
        ▼
 [Qwen LLM API Call]
 System prompt (rules + retrieved context)
 + conversation history (last 3 turns)
 + user message
        │
        ▼
 LLM generates answer (constrained by system prompt)
        │
        ▼
 Reply sent back to browser, history updated
```

---

## Requirements Checklist

1. **Only answer if question relates to data** — Handled by FAISS retrieval + system prompt rule #1.
2. **If answer not in data, say "I cannot answer that.."** — System prompt explicitly instructs this; no hallucination allowed.
3. **Answer greetings and self-introduction questions** — System prompt rule #2 permits this while blocking everything else.
4. **Resilient to jailbreak and injection** — Two-layer defense: regex pattern matching (17 patterns) blocks common attacks before the API call; system prompt rules #3–#4 reinforce this at the LLM level.
5. **Remember last N questions** — Conversation history (last 6 messages = 3 turns) is stored in Flask session and passed with every API call.
6. **Cannot manipulate answer format** — Regex detects format-control attempts ("answer in N words", "respond in X language"); system prompt rule #5 reinforces this.
7. **Smart FAISS chunking** — FAISS Index 2 uses TF-IDF bigrams to retrieve the top 5 most relevant text chunks from the full dataset for each query.
8. **Two FAISS models** — Index 1 (questions only, direct retrieval, no API call if score ≥ 0.75); Index 2 (all data, used as context for the LLM).
9. **Interface** — Flask web application with a Harry Potter-themed chat UI ("Pensieve"). Suggestion chips, typing indicator, session reset, Enter-to-send.

---

## Challenges

- **Threshold tuning for Index 1**: Finding the right similarity threshold (settled on 0.75) was tricky. Too low and irrelevant answers are returned directly; too high and the direct-answer shortcut is never triggered.
- **Balancing strictness**: Making the bot strict enough to refuse off-topic questions while still allowing greetings and self-introduction required careful prompt engineering. Simple phrases like "Who are you?" don't match any Harry Potter context, so the system prompt had to explicitly allow them.
- **Jailbreak resilience**: Regex-based detection catches common patterns but is inherently brittle. The solution was to use regex as a fast first pass and rely on the system prompt as a second, more flexible defense layer.
- **TF-IDF limitations**: TF-IDF cannot capture semantic similarity (e.g., "sorcerer" vs "wizard"). Phrase-level questions sometimes fail to match the correct chunk. Using bigrams helped but sentence-transformer embeddings would be more accurate.

## What We Enjoyed

- Building the dual-index retrieval system — it was satisfying to see exact question matches return answers instantly without an API call.
- Designing the chat UI with the Harry Potter aesthetic.
- Experimenting with prompt engineering to get the LLM to stay within strict constraints while still being natural.
