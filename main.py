from fastapi import FastAPI
import os
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer

# Setup (same as before)
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=20):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

text = "Leo is a fire sign ruled by the Sun and known for confidence, leadership, and warmth. Leos are natural performers who love attention and recognition."
chunks = chunk_text(text)

client = chromadb.Client()
collection = client.create_collection("astrology_notes")
collection.add(documents=chunks, ids=[f"chunk_{i}" for i in range(len(chunks))])

# This creates your "door"
app = FastAPI()

# This is the doorbell — people ring it by visiting /ask?question=...
@app.get("/ask")
def ask_question(question: str):
    results = collection.query(query_texts=[question], n_results=2)
    context = " ".join(results['documents'][0])
    prompt = f"Based on this context, summarize what it tells us: {context}\n\nQuestion: {question}"
    
    gemini_model = genai.GenerativeModel("gemini-3.8-flash")
    response = gemini_model.generate_content(prompt)
    
    return {"question": question, "answer": response.text}