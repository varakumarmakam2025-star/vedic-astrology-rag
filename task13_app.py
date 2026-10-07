import time
import streamlit as st
import os
import pandas as pd
import chromadb
from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai.errors import ServerError

# Load environment variables
load_dotenv(find_dotenv())

gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    st.error("GEMINI_API_KEY missing in .env file!")
    st.stop()

client_llm = genai.Client(api_key=gemini_key)

@st.cache_resource
def init_rag_system():
    text_data = ""
    if os.path.exists("astro_notes.txt"):
        with open("astro_notes.txt", "r", encoding="utf-8") as f:
            text_data = f.read()

    words = text_data.split()
    chunks = []
    chunk_size, overlap = 300, 50
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)

    client = chromadb.Client()
    collection = client.get_or_create_collection(name="astro_knowledge_ui")
    
    for idx, chunk in enumerate(chunks):
        collection.add(documents=[chunk], ids=[f"doc_{idx}"])
        
    return collection

collection = init_rag_system()

# UI Layout
st.title("Astro RAG Assistant")
st.write("Ask questions based on your astrology notes.")

query = st.text_input("Enter your question:", placeholder="e.g., What are the traits of Leo?")

def generate_with_retry(prompt):
    """Tries primary model with retry, then falls back to gemini-2.5-flash."""
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
                time.sleep(2)  # Wait 2 seconds before retry
            except Exception as e:
                st.error(f"Error calling model {model_name}: {e}")
                return None
    st.error("Google Gemini servers are currently overloaded. Please try again in a few moments.")
    return None

if st.button("Search & Answer") and query:
    with st.spinner("Retrieving context and generating answer..."):
        results = collection.query(query_texts=[query], n_results=2)
        retrieved_chunks = results['documents'][0]
        context = "\n\n".join(retrieved_chunks)

        prompt = f"""
        Answer the question based ONLY on the provided context below.

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
                for chunk in retrieved_chunks:
                    st.info(chunk)