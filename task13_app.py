import os
import time
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai.errors import ServerError

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

# Reuse the data and indexing code from task12 (one copy of the logic)
from task12_rag_pipeline import load_data, csv_rows_to_texts, chunk_text, setup_vector_db

client_llm = genai.Client(api_key=key)


@st.cache_resource(show_spinner="Building the knowledge base...")
def init_rag_system():
    raw_text, csv_data = load_data()
    chunks = chunk_text(raw_text)
    csv_texts = csv_rows_to_texts(csv_data)
    return setup_vector_db(chunks, csv_texts), len(chunks), len(csv_texts)


def retrieve(query, collection, top_k=3):
    results = collection.query(query_texts=[query], n_results=top_k)
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return [f"[{s}] {c}" for s, c in zip(sources, chunks)]


def generate_with_retry(prompt):
    """Tries the primary model with retries, then falls back to gemini-2.5-flash."""
    models_to_try = ["gemini-3.8-flash", "gemini-2.5-flash"]
    for model_name in models_to_try:
        for attempt in range(3):
            try:
                response = client_llm.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                return response.text
            except ServerError:
                time.sleep(2)
            except Exception as e:
                st.error(f"Error calling model {model_name}: {e}")
                return None
    st.error("Google Gemini servers are currently overloaded. Please try again in a few moments.")
    return None


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
        retrieved = retrieve(query, collection)
        context = "\n\n".join(retrieved)

        prompt = f"""
        Answer the question based ONLY on the provided context below.
        If the context does not contain the answer, say so.

        Context:
        {context}

        Question: {query}
        Answer:
        """

        answer = generate_with_retry(prompt)

        if answer:
            st.subheader("Answer:")
            st.write(answer)
            with st.expander("View Retrieved Context Chunks"):
                for chunk in retrieved:
                    st.info(chunk)