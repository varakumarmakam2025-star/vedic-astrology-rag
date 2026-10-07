import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=5):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


text = "Leo is a fire sign ruled by the Sun and known for confidence"
chunks = chunk_text(text)

client = chromadb.Client()
collection = client.create_collection("astrology_notes")

collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

results = collection.query(query_texts=["What does Leo represent?"], n_results=2)
print(results)