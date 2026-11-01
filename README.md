@'
# Astro RAG Assistant

A small retrieval-augmented generation (RAG) app that answers questions about Vedic astrology
from my own notes and a structured CSV, and shows which source each answer came from.

**Live demo:** https://vedic-astrology-rag.streamlit.app

## What it does
- Splits my notes (`astro_notes.txt`) into sentence-level chunks, each prefixed with its subject.
- Turns each row of `signs.csv` into a sentence and indexes it with the notes in ChromaDB.
- Retrieves the 3 closest chunks for a question and sends them to Gemini with the instruction
  to answer only from that context.
- Shows the retrieved chunks with source labels, plus timing for retrieval and generation.
- Tries several Gemini models in order with a 12 s timeout, so one overloaded model does not block answers.

## Evaluation
I wrote 20 test questions and measured recall@3: whether the text that must appear in the answer
is among the 3 retrieved chunks.

| Method | Recall@3 | Avg words sent to the LLM |
|---|---|---|
| Fixed-size chunks (300 words, 50 overlap) | 90% | 124 |
| Paragraph chunks | 75% | 52 |
| Sentence chunks with subject label (used in the app) | 85% | 35 |
| Hybrid (sentence chunks + keyword + vector search) | 90% | 34 |

What I learned:
- My first 10-question test ranked the methods differently from the 20-question test,
  so small test sets are misleading.
- Fixed-size chunks scored well partly because my notes are tiny, so one chunk contains
  everything. That also means sending the LLM much more text.
- Sentence chunking fixed a failure on "What does Mars represent?".
- Hybrid search is only used in `eval_hybrid.py`, not in the app.

## Limitations
- Tiny knowledge base: about 175 words of notes plus 12 sign rows.
- The test questions were partly drafted with AI assistance, and I chose methods after seeing
  failures on the same questions, so the scores are optimistic. The check is a keyword match,
  which can pass by luck.
- No reranking, no automated tests, and the vector database is rebuilt in memory on every start.
- The notes are my own summaries of traditional Vedic associations, not scientific claims.
- Gemini speed and availability vary, so response time varies.

## Run it locally
1. `pip install -r requirements.txt`
2. Create a `.env` file containing `GEMINI_API_KEY=your_key`
3. `streamlit run app.py`

On Streamlit Cloud the key goes in the app's Secrets as `GEMINI_API_KEY = "your_key"`.

## Stack
Python 3.12, ChromaDB, Gemini API (google-genai 2.33.0), Streamlit, pandas, Streamlit Community Cloud.

## How I built it
I built this to learn RAG from scratch, using AI coding assistants for help. I wrote the notes,
ran the experiments, and debugged the deployment.
'@ | Set-Content -Path README.md -Encoding ascii