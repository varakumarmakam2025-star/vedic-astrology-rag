import os
import pandas as pd
import chromadb
from dotenv import load_dotenv, find_dotenv
from google import genai

# Load environment variables
load_dotenv(find_dotenv())

gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    raise ValueError("GEMINI_API_KEY is missing! Check your .env file.")

# Initialize Gemini Client
client_llm = genai.Client(api_key=gemini_key)

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

# --- STEP 2: CHUNKING ---
def chunk_text(text, chunk_size=300, overlap=50):
    """Splits text into overlapping chunks."""
    words = text.split()
    if not words:
        return ["No content found in input files."]
    
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

# --- STEP 3: EMBEDDINGS & VECTOR STORE ---
def setup_vector_db(chunks):
    """Initializes ChromaDB vector store with chunk embeddings."""
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="astro_knowledge")
    
    for idx, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            ids=[f"doc_{idx}"]
        )
    return collection

def query_rag_pipeline(query, collection, top_k=2):
    """Retrieves context and passes it to Gemini."""
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    retrieved_chunks = results['documents'][0]
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
    Answer the question based ONLY on the provided context below.

    Context:
    {context}

    Question: {query}
    Answer:
    """

    # Updated to gemini-3.8-flash
    response = client_llm.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
    )
    
    return response.text, retrieved_chunks

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("1. Loading Data...")
    raw_text, _ = load_data()
    
    print("2. Chunking Text...")
    chunks = chunk_text(raw_text)
    
    print("3. Indexing Chunks in Vector DB...")
    vector_store = setup_vector_db(chunks)
    
    print("\n--- RAG System Ready ---")
    
    user_query = "What are the characteristics of fire signs?"
    answer, context_used = query_rag_pipeline(user_query, vector_store)
    
    print(f"\nUser Query: {user_query}\n")
    print("Retrieved Context:")
    for c in context_used:
        print(f"- {c}\n")
    print(f"LLM Response:\n{answer}")