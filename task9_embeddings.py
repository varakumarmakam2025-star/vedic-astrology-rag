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

for chunk in chunks:
    embedding = model.encode(chunk)
    print(f"Chunk: {chunk}")
    print(f"Embedding length: {len(embedding)}")
    print()