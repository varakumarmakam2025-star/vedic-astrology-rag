import os
import re
import time
import random
import pandas as pd
import chromadb
from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError

load_dotenv(find_dotenv())

gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    raise ValueError("GEMINI_API_KEY is missing! Check your .env file.")

client_llm = genai.Client(api_key=gemini_key)

# One place to change the model name.
MODEL = "gemini-3.8-flash"

# "low" thinking = faster answers; enough when answering from retrieved context.
GEN_CONFIG = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_level="low"),
)


# --- STEP 1: LOAD DATA ---
def load_data():
    """Reads your raw data files."""
    text_data = ""
    if os.path.exists("astro_notes.txt"):
        with open("astro_notes.txt", "r", encoding="utf-8") as f:
            text_data = f.read()

    csv_data = []
    if os.path.exists("signs.csv"):
        df = pd.read_csv("signs.csv")
        csv_data = df.to_dict(orient="records")

    return text_data, csv_data


def csv_rows_to_texts(csv_data):
    texts = []
    for row in csv_data:
        sign = str(row.get("sign", "")).strip()
        element = str(row.get("element", "")).strip()
        planet = str(row.get("ruling_planet", "")).strip()
        traits = str(row.get("traits", "")).strip()
        if sign and element and planet:
            text = f"{sign} is a {element} sign ruled by {planet}."
            if traits and traits != "nan":
                text += f" Known traits of {sign}: {traits}."
        else:
            text = ". ".join(f"{k}: {v}" for k, v in row.items()
                             if str(v).strip() and str(v) != "nan") + "."
        texts.append(text)
    return texts


# --- STEP 2: CHUNKING ---
def chunk_text(text, chunk_size=300, overlap=50):
    """Old method: fixed-size word chunks with overlap (kept for comparison)."""
    words = text.split()
    if not words:
        return []
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_by_paragraph(text, max_words=150):
    """New method: one chunk per paragraph. Long paragraphs are cut by sentence."""
    chunks = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        if len(paragraph.split()) <= max_words:
            chunks.append(paragraph)
        else:
            current = ""
            sentences = paragraph.replace(". ", ".|").replace("! ", "!|").replace("? ", "?|").split("|")
            for sentence in sentences:
                if current and len((current + " " + sentence).split()) > max_words:
                    chunks.append(current.strip())
                    current = sentence
                else:
                    current = (current + " " + sentence).strip()
            if current:
                chunks.append(current.strip())
    return chunks


# --- STEP 3: VECTOR STORE ---
def setup_vector_db(note_chunks, csv_texts):
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="astro_knowledge")

    docs, ids, metas = [], [], []
    for i, chunk in enumerate(note_chunks):
        docs.append(chunk)
        ids.append(f"note_{i}")
        metas.append({"source": "astro_notes.txt"})
    for i, text in enumerate(csv_texts):
        docs.append(text)
        ids.append(f"csv_{i}")
        metas.append({"source": "signs.csv"})

    if docs:
        collection.add(documents=docs, ids=ids, metadatas=metas)
    return collection


def generate_with_retry(client, tries=3, **kwargs):
    """Retries only when Google's servers are overloaded.

    Other errors (wrong model name, bad key) fail immediately instead of
    sleeping and retrying for no reason.
    """
    kwargs.setdefault("config", GEN_CONFIG)
    for attempt in range(tries):
        try:
            return client.models.generate_content(**kwargs)
        except ServerError as e:
            print(f"Attempt {attempt + 1} failed (server overloaded): {e}")
            if attempt < tries - 1:
                time.sleep((2 ** attempt) + random.uniform(0, 1))
        except Exception as e:
            print(f"Model call failed: {e}")
            return None
    return None


def query_rag_pipeline(query, collection, top_k=3):
    """Retrieves context (with source labels) and passes it to Gemini."""
    results = collection.query(query_texts=[query], n_results=top_k)
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    labeled = [f"[{s}] {c}" for s, c in zip(sources, chunks)]
    context = "\n\n".join(labeled)

    prompt = f"""
    Answer the question based ONLY on the provided context below.
    If the context does not contain the answer, say so.

    Context:
    {context}

    Question: {query}
    Answer:
    """

    response = generate_with_retry(client_llm, model=MODEL, contents=prompt)
    if response is None:
        return "Sorry, the model did not respond. Please try again.", labeled
    return response.text, labeled


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("1. Loading Data...")
    raw_text, csv_data = load_data()
    if "___" in raw_text:
        print("WARNING: astro_notes.txt still contains ___ blanks. Replace them with your own text.")

    print("2. Chunking Text...")
    chunks = chunk_by_paragraph(raw_text)
    csv_texts = csv_rows_to_texts(csv_data)
    print("Words in notes:", len(raw_text.split()), "| Note chunks:", len(chunks), "| CSV rows:", len(csv_texts))

    print("3. Indexing in Vector DB...")
    vector_store = setup_vector_db(chunks, csv_texts)

    print("\n--- RAG System Ready ---")

    user_query = "What are fire signs like?"
    answer, context_used = query_rag_pipeline(user_query, vector_store)

    print(f"\nUser Query: {user_query}\n")
    print("Retrieved Context:")
    for c in context_used:
        print(f"- {c}\n")
    print(f"LLM Response:\n{answer}")