import os
import re
import time
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai import types

load_dotenv(find_dotenv())
st.set_page_config(page_title="Astro RAG Assistant")

# API key: .env on your laptop, or Streamlit Secrets in the cloud
key = os.getenv("GEMINI_API_KEY")
if not key:
    try:
        key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        key = None
if not key:
    st.error("GEMINI_API_KEY is missing. Add it to your .env file (local) "
             "or to the app's Secrets settings (Streamlit Cloud).")
    st.stop()
os.environ["GEMINI_API_KEY"] = key

from task12_rag_pipeline import load_data, csv_rows_to_texts, setup_vector_db

# Models measured as working on 2026-10-29, fastest first. The last one is
# higher quality but was overloaded, so it is only a last resort.
MODELS = ["gemini-3.1-flash-lite", "gemini-flash-lite-latest",
          "gemini-3.5-flash-lite", "gemini-3.8-flash"]

client_llm = genai.Client(api_key=key,
                          http_options=types.HttpOptions(timeout=12000))


def chunk_by_sentence_with_label(text):
    """One chunk per sentence, prefixed with its paragraph's subject."""
    chunks = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        label = paragraph.split(" (")[0] if " (" in paragraph else paragraph.split(" ")[0]
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            if sentence.strip():
                chunks.append(f"{label}: {sentence.strip()}")
    return chunks


@st.cache_resource(show_spinner="Building the knowledge base...")
def init_rag_system():
    raw_text, csv_data = load_data()
    chunks = chunk_by_sentence_with_label(raw_text)
    csv_texts = csv_rows_to_texts(csv_data)
    return setup_vector_db(chunks, csv_texts), len(chunks), len(csv_texts)


def retrieve(query, collection, top_k=3):
    results = collection.query(query_texts=[query], n_results=top_k)
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return [f"[{s}] {c}" for s, c in zip(sources, chunks)]


def generate_with_fallback(prompt):
    """Tries each model once with a 12 s timeout, then moves on immediately."""
    errors = []
    for model_name in MODELS:
        start = time.perf_counter()
        try:
            response = client_llm.models.generate_content(model=model_name, contents=prompt)
            return response.text, model_name, time.perf_counter() - start, errors
        except Exception as e:
            errors.append(f"{model_name}: {str(e)[:60]}")
    return None, None, 0.0, errors


collection, n_chunks, n_rows = init_rag_system()

# UI Layout
st.title("Astro RAG Assistant")
st.write("Ask questions based on the astrology notes in this project.")
st.caption(f"Knowledge base: {n_chunks} note chunk(s) and {n_rows} sign rows.")

examples = ["Which signs are fire signs?", "What is Aries like?", "What does Mars represent?"]
cols = st.columns(len(examples))
for col, example in zip(cols, examples):
    if col.button(example):
        st.session_state["question"] = example

query = st.text_input("Enter your question:", key="question",
                      placeholder="e.g., What are the traits of Leo?")

if st.button("Search & Answer") and query.strip():
    with st.spinner("Retrieving context and generating answer..."):
        t0 = time.perf_counter()
        retrieved = retrieve(query, collection)
        retrieval_s = time.perf_counter() - t0
        context = "\n\n".join(retrieved)

        prompt = f"""
        Answer the question based ONLY on the provided context below.
        If the context does not contain the answer, say so.

        Context:
        {context}

        Question: {query}
        Answer:
        """

        answer, model_used, gen_s, errors = generate_with_fallback(prompt)

    if answer:
        st.subheader("Answer:")
        st.write(answer)
        st.caption(f"Retrieval {retrieval_s:.2f}s | {model_used} {gen_s:.1f}s")
        with st.expander("View Retrieved Context Chunks"):
            for chunk in retrieved:
                st.info(chunk)
    else:
        st.error("All models are unavailable right now. Please try again in a minute.")
        with st.expander("Details"):
            for err in errors:
                st.write(err)