import os
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer

# Load your API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Reuse your existing chunking + ChromaDB setup
model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=20):  # bumped up chunk size for better context
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

text = "Leo is a fire sign ruled by the Sun and known for confidence, leadership, and warmth. Leos are natural performers who love attention and recognition."
chunks = chunk_text(text)

client = chromadb.Client()
collection = client.create_collection("astrology_notes")
collection.add(documents=chunks, ids=[f"chunk_{i}" for i in range(len(chunks))])

# Retrieve relevant chunks for a question
question = "What does Leo represent?"
results = collection.query(query_texts=[question], n_results=2)
retrieved_chunks = results['documents'][0]

# Build the prompt with retrieved context
context = " ".join(retrieved_chunks)
prompt = f"Based on this context, summarize what it tells us: {context}\n\nQuestion: {question}"

# Send to Gemini
gemini_model = genai.GenerativeModel("gemini-3.8-flash")
response = gemini_model.generate_content(prompt)

print("Question:", question)
print("\nRetrieved context:", context)
print("\nGemini's answer:", response.text)